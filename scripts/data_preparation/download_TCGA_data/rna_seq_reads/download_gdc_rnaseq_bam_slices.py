#!/usr/bin/env python3

"""
Download RNA-seq BAM gene slices from GDC using a GDC manifest.

Typical use:
  python rna_seq_reads/download_gdc_rnaseq_bam_slices.py

The gene file should contain one HGNC/GENCODE gene symbol per line:
  PTEN
  TP53
  CDKN2A

Notes:
  - For gene-based slicing, use genomic RNA-seq BAMs, e.g. STAR 2-Pass Genome BAMs.
  - Transcriptome BAMs often use transcript reference names such as ENST..., so genomic
    gene slicing may not work as expected.
  - Controlled-access BAMs require a valid GDC token.
"""

import argparse
import csv
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

import requests
from tqdm.auto import tqdm


GDC_SLICING_URL = "https://api.gdc.cancer.gov/slicing/view/{file_id}"
DEFAULT_MANIFEST = Path("rna_seq_reads/gdc_manifest.2026-06-12.143040.txt")
DEFAULT_TOKEN = Path(
    "rna_seq_reads/.gdc-user-token.2026-06-12T19_38_20.239Z.txt"
)
DEFAULT_GENES_FILE = Path("rna_seq_reads/pancancer_1pc_genes.txt")
DEFAULT_OUTDIR = Path("rna_seq_reads/gdc_rnaseq_gene_slices")
SAMTOOLS = "samtools"
TIMEOUT_SECONDS = 300
MAX_ATTEMPTS = 5
RETRY_WAIT_SECONDS = 30
NUM_WORKERS = 4


def read_manifest(manifest_path: Path) -> List[Dict[str, str]]:
    """Read a GDC manifest TSV."""
    with manifest_path.open("r", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def safe_filename(name: str) -> str:
    """
    Make a filesystem-safe filename component.
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def load_token(token_path: Path) -> str:
    """Read GDC token file."""
    return token_path.read_text().strip()


def read_genes_file(genes_path: Path) -> List[str]:
    """Read one gene symbol per line, ignoring blank lines and comments."""
    genes = {
        line.strip()
        for line in genes_path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    return sorted(genes)


def download_bam_slice_for_file(
    file_id: str,
    filename: str,
    genes: List[str],
    token: str,
    out_bam: Path,
) -> None:
    """Download one BAM slice from the GDC slicing API."""
    url = GDC_SLICING_URL.format(file_id=file_id)
    headers = {
        "X-Auth-Token": token,
        "Content-Type": "application/json",
    }

    out_bam.parent.mkdir(parents=True, exist_ok=True)
    tmp_bam = out_bam.with_suffix(out_bam.suffix + ".partial")
    print(
        f"[INFO] Slicing {filename} ({file_id}) for genes: {','.join(genes)}",
        flush=True,
    )

    # Retry transient GDC server and network failures.
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            tmp_bam.unlink(missing_ok=True)
            with requests.post(
                url,
                headers=headers,
                json={"gencode": genes},
                stream=True,
                timeout=TIMEOUT_SECONDS,
            ) as response:
                if not response.ok:
                    detail = response.text[:500].strip()
                    raise requests.HTTPError(
                        f"HTTP {response.status_code} from GDC: {detail}",
                        response=response,
                    )
                total_bytes = int(response.headers.get("Content-Length", 0)) or None
                with (
                    tmp_bam.open("wb") as handle,
                    tqdm(
                        total=total_bytes,
                        desc=f"Downloading {safe_filename(filename)[:40]}",
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        leave=True,
                    ) as byte_progress,
                ):
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        handle.write(chunk)
                        byte_progress.update(len(chunk))
            break
        except requests.RequestException as error:
            if attempt == MAX_ATTEMPTS:
                raise
            wait_seconds = RETRY_WAIT_SECONDS * attempt
            tqdm.write(
                f"[WARN] Download failed for {file_id}: {error}; "
                f"retrying in {wait_seconds}s ({attempt}/{MAX_ATTEMPTS})"
            )
            time.sleep(wait_seconds)

    # Publish the BAM only after the download finishes successfully.
    tmp_bam.replace(out_bam)
    print(f"[OK] Wrote {out_bam}", flush=True)


def index_bam(bam_path: Path) -> None:
    """Index a BAM slice using samtools."""
    print(f"[INFO] Indexing {bam_path}", flush=True)
    subprocess.run([SAMTOOLS, "index", str(bam_path)], check=True)


def bam_index_paths(bam_path: Path) -> tuple[Path, Path]:
    """Return the two common samtools BAM index locations."""
    return Path(f"{bam_path}.bai"), bam_path.with_suffix(".bai")


def bam_passes_quickcheck(bam_path: Path) -> tuple[bool, str]:
    """Return whether samtools considers a BAM complete and readable."""
    if not bam_path.exists() or bam_path.stat().st_size == 0:
        return False, "file is missing or empty"
    result = subprocess.run(
        [SAMTOOLS, "quickcheck", "-v", str(bam_path)],
        capture_output=True,
        text=True,
    )
    detail = (result.stderr or result.stdout).strip()
    return result.returncode == 0, detail


def remove_bam_and_indexes(bam_path: Path) -> None:
    """Remove an invalid BAM and any indexes derived from it."""
    bam_path.unlink(missing_ok=True)
    for index_path in bam_index_paths(bam_path):
        index_path.unlink(missing_ok=True)


def has_current_index(bam_path: Path) -> bool:
    """Return whether a non-empty BAM index is at least as new as the BAM."""
    for index_path in bam_index_paths(bam_path):
        if (
            index_path.exists()
            and index_path.stat().st_size > 0
            and index_path.stat().st_mtime >= bam_path.stat().st_mtime
        ):
            return True
    return False


def process_manifest_row(
    row: Dict[str, str],
    genes: List[str],
    token: str,
    outdir: Path,
    gene_label: str,
) -> None:
    """Download and index one manifest entry."""
    file_id = row["id"]
    filename = row.get("filename", file_id)
    base = safe_filename(filename.replace(".bam", ""))
    out_bam = outdir / file_id / f"{base}.{gene_label}.slice.bam"

    bam_is_valid, quickcheck_detail = bam_passes_quickcheck(out_bam)
    if bam_is_valid:
        tqdm.write(f"[SKIP] Valid BAM exists: {out_bam}")
    else:
        if out_bam.exists():
            detail = quickcheck_detail or "samtools quickcheck failed"
            tqdm.write(
                f"[WARN] Invalid existing BAM; redownloading {out_bam}: "
                f"{detail}"
            )
            remove_bam_and_indexes(out_bam)
        download_bam_slice_for_file(
            file_id=file_id,
            filename=filename,
            genes=genes,
            token=token,
            out_bam=out_bam,
        )
        bam_is_valid, quickcheck_detail = bam_passes_quickcheck(out_bam)
        if not bam_is_valid:
            remove_bam_and_indexes(out_bam)
            raise RuntimeError(
                f"Downloaded BAM failed samtools quickcheck: {out_bam}: "
                f"{quickcheck_detail or 'unknown validation error'}"
            )

    if has_current_index(out_bam):
        tqdm.write(f"[SKIP] Current BAM index exists: {out_bam}")
    else:
        for index_path in bam_index_paths(out_bam):
            index_path.unlink(missing_ok=True)
        index_bam(out_bam)


def main() -> None:
    # Parse input paths, using the project-specific defaults above.
    parser = argparse.ArgumentParser(
        description="Download GDC RNA-seq BAM gene slices from a manifest."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"GDC manifest TSV file (default: {DEFAULT_MANIFEST}).",
    )
    parser.add_argument(
        "--token",
        type=Path,
        default=DEFAULT_TOKEN,
        help=f"GDC access token file (default: {DEFAULT_TOKEN}).",
    )
    parser.add_argument(
        "--genes-file",
        type=Path,
        default=DEFAULT_GENES_FILE,
        help=f"Gene symbol file (default: {DEFAULT_GENES_FILE}).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help=f"Output directory (default: {DEFAULT_OUTDIR}).",
    )
    args = parser.parse_args()

    # Load the manifest and prepare the output directory.
    rows = read_manifest(args.manifest)
    print(f"[INFO] Manifest rows: {len(rows)}", flush=True)
    print(f"[INFO] Output directory: {args.outdir}", flush=True)
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Load the requested genes and GDC authentication token.
    genes = read_genes_file(args.genes_file)
    print(
        f"[INFO] Loaded {len(genes)} unique gene symbol(s) from "
        f"{args.genes_file}",
        flush=True,
    )

    token = load_token(args.token)

    # Build a short gene label for the output BAM filenames.
    gene_label = safe_filename("_".join(genes))
    if len(gene_label) > 120:
        gene_label = safe_filename(f"{len(genes)}genes")

    # Download and index manifest entries in parallel.
    print(f"[INFO] Parallel workers: {NUM_WORKERS}", flush=True)
    failed_file_ids = []
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(
                process_manifest_row,
                row,
                genes,
                token,
                args.outdir,
                gene_label,
            ): row["id"]
            for row in rows
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="BAM slices",
            unit="file",
        ):
            file_id = futures[future]
            try:
                future.result()
            except Exception as error:
                failed_file_ids.append(file_id)
                tqdm.write(f"[ERROR] Skipping {file_id}: {error}")

    if failed_file_ids:
        print(f"[WARN] Failed file IDs: {', '.join(failed_file_ids)}")


if __name__ == "__main__":
    main()
