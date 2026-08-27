#!/usr/bin/env python3

"""
For each BAM file, validate and index it, convert it to BigWig, and then
prepare its fixed-size patient gene tensor before moving to the next file.
Multiple files are still processed concurrently across workers.

BigWig files represent genomic coverage; they are not lossless BAM
conversions. This script uses:

  - samtools quickcheck/index for BAM validation and indexing
  - deepTools bamCoverage for BigWig generation
  - pyBigWig for extracting hierarchical gene coverage tensors

Examples:
  python rna_seq_reads/index_bams_and_convert_to_bigwig.py

  python rna_seq_reads/index_bams_and_convert_to_bigwig.py \
      --input-dir rna_seq_reads/gdc_rnaseq_gene_slices \
      --output-dir rna_seq_reads/gdc_rnaseq_gene_bigwigs \
      --bin-size 10 \
      --normalize-using CPM

By default, coverage is not normalized. For GDC gene-sliced BAMs, CPM/RPKM
normalization uses only reads present in the slice, not the original whole
RNA-seq library, so normalized values are not equivalent to whole-library
normalization.

Tensor outputs contain log1p coverage in 64-bp bins grouped into 256-bin
(16,384-bp) chunks. Negative-strand genes are reversed into transcriptional
orientation.
"""

import argparse
import json
import math
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyBigWig
from tqdm.auto import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
DEFAULT_INPUT_DIR = SCRIPT_DIR / "gdc_rnaseq_gene_slices"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "gdc_rnaseq_gene_bigwigs"
DEFAULT_SAMPLE_SHEET = SCRIPT_DIR / "gdc_sample_sheet.2026-06-12.tsv"
DEFAULT_GENES = REPO_ROOT / "data/driver_genes_coords/Pancancer_1pc.tsv"
DEFAULT_TENSOR_OUTPUT_DIR = REPO_ROOT / "data/patients_bigwig_gene_tensors"
DEFAULT_WORKERS = 4
DEFAULT_THREADS_PER_FILE = 2
DEFAULT_TENSOR_BIN_BP = 64
DEFAULT_CHUNK_BINS = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Index BAM files, generate BigWigs, and prepare patient gene "
            "coverage tensors."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing BAM files (default: {DEFAULT_INPUT_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"BigWig output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--bin-size",
        type=int,
        default=10,
        help="BigWig bin size in base pairs (default: 10).",
    )
    parser.add_argument(
        "--normalize-using",
        choices=["None", "CPM", "RPKM", "BPM", "RPGC"],
        default="None",
        help="bamCoverage normalization method (default: None).",
    )
    parser.add_argument(
        "--effective-genome-size",
        type=int,
        default=None,
        help="Required by bamCoverage when --normalize-using RPGC.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of BAM files processed concurrently (default: {DEFAULT_WORKERS}).",
    )
    parser.add_argument(
        "--threads-per-file",
        type=int,
        default=DEFAULT_THREADS_PER_FILE,
        help=(
            "Threads passed to samtools and bamCoverage for each file "
            f"(default: {DEFAULT_THREADS_PER_FILE})."
        ),
    )
    parser.add_argument(
        "--minimum-mapping-quality",
        type=int,
        default=0,
        help="Exclude reads below this mapping quality (default: 0).",
    )
    parser.add_argument(
        "--ignore-duplicates",
        action="store_true",
        help="Tell bamCoverage to ignore duplicate reads.",
    )
    parser.add_argument(
        "--skip-non-covered-regions",
        action="store_true",
        help="Omit zero-coverage regions from BigWig output.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate existing BigWig files.",
    )
    parser.add_argument(
        "--sample-sheet",
        type=Path,
        default=DEFAULT_SAMPLE_SHEET,
        help="GDC sample sheet used to map File IDs to TCGA Case IDs.",
    )
    parser.add_argument(
        "--genes",
        type=Path,
        default=DEFAULT_GENES,
        help="Gene coordinate TSV containing gene, chr, strand, start, and end.",
    )
    parser.add_argument(
        "--tensor-output-dir",
        type=Path,
        default=DEFAULT_TENSOR_OUTPUT_DIR,
        help="Directory for per-patient compressed NPZ tensors.",
    )
    parser.add_argument(
        "--tensor-bin-bp",
        type=int,
        default=DEFAULT_TENSOR_BIN_BP,
        help="Base pairs represented by one tensor position (default: 64).",
    )
    parser.add_argument(
        "--chunk-bins",
        type=int,
        default=DEFAULT_CHUNK_BINS,
        help="Tensor positions per genomic chunk (default: 256).",
    )
    parser.add_argument(
        "--force-tensors",
        action="store_true",
        help="Regenerate existing patient tensor files.",
    )
    parser.add_argument(
        "--skip-tensors",
        action="store_true",
        help="Stop after BigWig generation without preparing tensors.",
    )
    return parser.parse_args()


def require_executable(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise FileNotFoundError(
            f"Required executable not found in PATH: {name}"
        )
    return path


def discover_bams(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*.bam")
        if not path.name.endswith(".partial.bam")
    )


def bam_index_paths(bam_path: Path) -> tuple[Path, Path]:
    return (
        Path(f"{bam_path}.bai"),
        bam_path.with_suffix(".bai"),
    )


def has_current_index(bam_path: Path) -> bool:
    for index_path in bam_index_paths(bam_path):
        if (
            index_path.exists()
            and index_path.stat().st_size > 0
            and index_path.stat().st_mtime >= bam_path.stat().st_mtime
        ):
            return True
    return False


def validate_bam(bam_path: Path, samtools: str) -> None:
    result = subprocess.run(
        [samtools, "quickcheck", "-v", str(bam_path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(
            f"samtools quickcheck failed for {bam_path}: {detail}"
        )


def index_bam(
    bam_path: Path,
    samtools: str,
    threads_per_file: int,
) -> None:
    if has_current_index(bam_path):
        return
    command = [
        samtools,
        "index",
        "-@",
        str(threads_per_file),
        str(bam_path),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(
            f"samtools index failed for {bam_path}: {detail}"
        )


def output_bigwig_path(
    bam_path: Path,
    input_dir: Path,
    output_dir: Path,
) -> Path:
    relative_path = bam_path.relative_to(input_dir)
    return (output_dir / relative_path).with_suffix(".bw")


def run_bam_coverage(
    bam_path: Path,
    bigwig_path: Path,
    bam_coverage: str,
    args: argparse.Namespace,
) -> None:
    if (
        bigwig_path.exists()
        and bigwig_path.stat().st_size > 0
        and not args.force
    ):
        return

    bigwig_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = bigwig_path.with_suffix(".bw.partial")
    partial_path.unlink(missing_ok=True)

    command = [
        bam_coverage,
        "--bam",
        str(bam_path),
        "--outFileName",
        str(partial_path),
        "--outFileFormat",
        "bigwig",
        "--binSize",
        str(args.bin_size),
        "--numberOfProcessors",
        str(args.threads_per_file),
        "--minMappingQuality",
        str(args.minimum_mapping_quality),
    ]
    if args.normalize_using != "None":
        command.extend(["--normalizeUsing", args.normalize_using])
    if args.normalize_using == "RPGC":
        command.extend(
            [
                "--effectiveGenomeSize",
                str(args.effective_genome_size),
            ]
        )
    if args.ignore_duplicates:
        command.append("--ignoreDuplicates")
    if args.skip_non_covered_regions:
        command.append("--skipNonCoveredRegions")

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(
                f"bamCoverage failed for {bam_path}: {detail}"
            )
        partial_path.replace(bigwig_path)
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise


def process_bam(
    bam_path: Path,
    input_dir: Path,
    output_dir: Path,
    samtools: str,
    bam_coverage: str,
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    validate_bam(bam_path, samtools)
    index_bam(bam_path, samtools, args.threads_per_file)
    bigwig_path = output_bigwig_path(
        bam_path,
        input_dir,
        output_dir,
    )
    run_bam_coverage(
        bam_path,
        bigwig_path,
        bam_coverage,
        args,
    )
    return bam_path, bigwig_path


def load_genes(path: Path) -> pd.DataFrame:
    genes = pd.read_csv(path, sep="\t")
    required = {"gene", "chr", "strand", "start", "end"}
    missing = required - set(genes.columns)
    if missing:
        raise ValueError(
            f"Missing gene columns in {path}: {sorted(missing)}"
        )
    genes = genes.drop_duplicates("gene", keep="first").copy()
    genes["gene"] = genes["gene"].astype(str)
    genes["chr"] = genes["chr"].astype(str)
    genes["strand"] = genes["strand"].astype(str)
    genes["start"] = genes["start"].astype(int)
    genes["end"] = genes["end"].astype(int)
    invalid = genes["end"] < genes["start"]
    if invalid.any():
        raise ValueError(
            f"Invalid gene intervals: "
            f"{genes.loc[invalid, 'gene'].tolist()}"
        )
    return genes.reset_index(drop=True)


def discover_bigwigs(bigwig_dir: Path) -> dict[str, Path]:
    paths = sorted(
        list(bigwig_dir.rglob("*.bw"))
        + list(bigwig_dir.rglob("*.bigWig"))
    )
    mapping = {}
    for path in paths:
        file_id = path.parent.name
        if file_id in mapping:
            raise ValueError(
                f"Multiple BigWigs found for GDC File ID {file_id}: "
                f"{mapping[file_id]} and {path}"
            )
        mapping[file_id] = path
    return mapping


def map_cases_to_bigwigs(
    sample_sheet_path: Path,
    bigwigs: dict[str, Path],
) -> tuple[dict[str, tuple[str, Path]], list[dict]]:
    sample_sheet = pd.read_csv(
        sample_sheet_path, sep="\t", dtype=str
    )
    required = {"File ID", "Case ID"}
    missing = required - set(sample_sheet.columns)
    if missing:
        raise ValueError(
            f"Missing sample-sheet columns: {sorted(missing)}"
        )
    rows = sample_sheet[
        sample_sheet["File ID"].isin(bigwigs)
    ].copy()
    if rows.empty:
        raise ValueError(
            "No BigWig File IDs matched the GDC sample sheet"
        )

    tissue_type = (
        rows["Tissue Type"]
        if "Tissue Type" in rows
        else pd.Series("", index=rows.index)
    )
    tumor_descriptor = (
        rows["Tumor Descriptor"]
        if "Tumor Descriptor" in rows
        else pd.Series("", index=rows.index)
    )
    rows["_tumor_rank"] = (
        tissue_type.fillna("").eq("Tumor").map({True: 0, False: 1})
    )
    rows["_primary_rank"] = (
        tumor_descriptor.fillna("").eq("Primary").map(
            {True: 0, False: 1}
        )
    )
    rows = rows.sort_values(
        ["Case ID", "_tumor_rank", "_primary_rank", "File ID"]
    )

    selected = {}
    duplicates = []
    for case_id, group in rows.groupby("Case ID", sort=True):
        chosen = group.iloc[0]
        file_id = str(chosen["File ID"])
        selected[str(case_id)] = (file_id, bigwigs[file_id])
        if len(group) > 1:
            duplicates.append(
                {
                    "case_id": str(case_id),
                    "selected_file_id": file_id,
                    "candidate_file_ids": (
                        group["File ID"].astype(str).tolist()
                    ),
                }
            )
    return selected, duplicates


def binned_gene_coverage(
    bigwig,
    chromosome: str,
    start_1based: int,
    end_1based: int,
    strand: str,
    bin_bp: int,
) -> np.ndarray:
    chromosomes = bigwig.chroms()
    if chromosome not in chromosomes:
        alternate = (
            chromosome.removeprefix("chr")
            if chromosome.startswith("chr")
            else f"chr{chromosome}"
        )
        if alternate not in chromosomes:
            raise ValueError(
                f"Chromosome {chromosome} not found in BigWig"
            )
        chromosome = alternate

    start = max(0, start_1based - 1)
    end = min(end_1based, int(chromosomes[chromosome]))
    if end <= start:
        return np.zeros(1, dtype=np.float32)

    n_bins = max(1, math.ceil((end - start) / bin_bp))
    values = bigwig.stats(
        chromosome,
        start,
        end,
        type="mean",
        nBins=n_bins,
    )
    coverage = np.asarray(
        [0.0 if value is None else value for value in values],
        dtype=np.float32,
    )
    coverage = np.nan_to_num(
        coverage, nan=0.0, posinf=0.0, neginf=0.0
    )
    coverage = np.clip(coverage, 0.0, None)
    if strand == "-":
        coverage = coverage[::-1].copy()
    return np.log1p(coverage)


def chunk_gene(
    coverage: np.ndarray,
    chunk_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_chunks = max(1, math.ceil(len(coverage) / chunk_bins))
    chunks = np.zeros((n_chunks, chunk_bins), dtype=np.float32)
    masks = np.zeros((n_chunks, chunk_bins), dtype=bool)
    for chunk_index in range(n_chunks):
        start = chunk_index * chunk_bins
        end = min(start + chunk_bins, len(coverage))
        width = end - start
        if width > 0:
            chunks[chunk_index, :width] = coverage[start:end]
            masks[chunk_index, :width] = True
    return chunks, masks


def process_patient_tensor(
    case_id: str,
    file_id: str,
    bigwig_path: Path,
    genes: pd.DataFrame,
    output_dir: Path,
    bin_bp: int,
    chunk_bins: int,
    force: bool,
) -> dict:
    output_path = output_dir / f"{case_id}.npz"
    if (
        output_path.exists()
        and output_path.stat().st_size > 0
        and not force
    ):
        return {
            "case_id": case_id,
            "file_id": file_id,
            "bigwig": str(bigwig_path),
            "output": str(output_path),
            "status": "skipped",
        }

    all_chunks = []
    all_masks = []
    offsets = [0]
    with pyBigWig.open(str(bigwig_path)) as bigwig:
        for row in genes.itertuples(index=False):
            coverage = binned_gene_coverage(
                bigwig,
                str(row.chr),
                int(row.start),
                int(row.end),
                str(row.strand),
                bin_bp,
            )
            chunks, masks = chunk_gene(coverage, chunk_bins)
            all_chunks.append(chunks)
            all_masks.append(masks)
            offsets.append(offsets[-1] + len(chunks))

    coverage_array = np.concatenate(all_chunks, axis=0).astype(
        np.float16, copy=False
    )
    mask_array = np.concatenate(all_masks, axis=0)
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".npz.partial")
    temporary_path.unlink(missing_ok=True)
    with temporary_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            coverage=coverage_array,
            position_mask=mask_array,
            gene_offsets=np.asarray(offsets, dtype=np.int32),
            gene_names=genes["gene"].to_numpy(dtype=str),
            case_id=np.asarray(case_id),
            gdc_file_id=np.asarray(file_id),
            source_bigwig=np.asarray(str(bigwig_path)),
            bin_bp=np.asarray(bin_bp, dtype=np.int32),
            chunk_bins=np.asarray(chunk_bins, dtype=np.int32),
        )
    temporary_path.replace(output_path)
    return {
        "case_id": case_id,
        "file_id": file_id,
        "bigwig": str(bigwig_path),
        "output": str(output_path),
        "status": "written",
        "total_chunks": int(coverage_array.shape[0]),
    }


def process_bam_pipeline(
    bam_path: Path,
    input_dir: Path,
    output_dir: Path,
    samtools: str,
    bam_coverage: str,
    args: argparse.Namespace,
    case_id: str | None,
    genes: pd.DataFrame | None,
) -> dict:
    _, bigwig_path = process_bam(
        bam_path,
        input_dir,
        output_dir,
        samtools,
        bam_coverage,
        args,
    )
    tensor_result = None
    if case_id is not None:
        if genes is None:
            raise ValueError("Gene coordinates are required for tensors")
        tensor_result = process_patient_tensor(
            case_id,
            bam_path.parent.name,
            bigwig_path,
            genes,
            args.tensor_output_dir,
            args.tensor_bin_bp,
            args.chunk_bins,
            args.force_tensors,
        )
    return {
        "bam": str(bam_path),
        "bigwig": str(bigwig_path),
        "case_id": case_id,
        "tensor": tensor_result,
    }


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    args.sample_sheet = args.sample_sheet.resolve()
    args.genes = args.genes.resolve()
    args.tensor_output_dir = args.tensor_output_dir.resolve()

    if not input_dir.is_dir():
        raise NotADirectoryError(
            f"Input directory not found: {input_dir}"
        )
    if args.bin_size < 1:
        raise ValueError("--bin-size must be >= 1")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.threads_per_file < 1:
        raise ValueError("--threads-per-file must be >= 1")
    if args.minimum_mapping_quality < 0:
        raise ValueError("--minimum-mapping-quality must be >= 0")
    if args.tensor_bin_bp < 1:
        raise ValueError("--tensor-bin-bp must be >= 1")
    if args.chunk_bins < 8:
        raise ValueError("--chunk-bins must be >= 8")
    if (
        args.normalize_using == "RPGC"
        and args.effective_genome_size is None
    ):
        raise ValueError(
            "--effective-genome-size is required with "
            "--normalize-using RPGC"
        )
    if not args.skip_tensors:
        if not args.sample_sheet.is_file():
            raise FileNotFoundError(
                f"Sample sheet not found: {args.sample_sheet}"
            )
        if not args.genes.is_file():
            raise FileNotFoundError(
                f"Gene coordinate file not found: {args.genes}"
            )

    samtools = require_executable("samtools")
    bam_coverage = require_executable("bamCoverage")
    bams = discover_bams(input_dir)
    if not bams:
        raise FileNotFoundError(
            f"No BAM files found recursively under {input_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    genes = None
    duplicates = []
    selected_case_by_file_id: dict[str, str] = {}
    if not args.skip_tensors:
        genes = load_genes(args.genes)
        args.tensor_output_dir.mkdir(parents=True, exist_ok=True)

        expected_bigwigs = {}
        for bam_path in bams:
            file_id = bam_path.parent.name
            if file_id in expected_bigwigs:
                raise ValueError(
                    f"Multiple BAMs found for GDC File ID {file_id}; "
                    "cannot choose one tensor source"
                )
            expected_bigwigs[file_id] = output_bigwig_path(
                bam_path, input_dir, output_dir
            )
        case_mapping, duplicates = map_cases_to_bigwigs(
            args.sample_sheet, expected_bigwigs
        )
        selected_case_by_file_id = {
            file_id: case_id
            for case_id, (file_id, _) in case_mapping.items()
        }

    print(f"[INFO] Input directory: {input_dir}")
    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] BAM files: {len(bams)}")
    print(
        f"[INFO] Workers: {args.workers}; "
        f"threads per file: {args.threads_per_file}"
    )
    print(f"[INFO] Normalization: {args.normalize_using}")
    if args.skip_tensors:
        print("[INFO] Tensor preparation: skipped")
    else:
        print(f"[INFO] Tensor genes: {len(genes)}")
        print(
            f"[INFO] Tensor cases selected: "
            f"{len(selected_case_by_file_id)}"
        )
        print(
            f"[INFO] Per-file pipeline: BAM -> BigWig -> tensor; "
            f"{args.chunk_bins} x {args.tensor_bin_bp} bp bins"
        )

    failures: list[tuple[Path, str]] = []
    pipeline_results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_bam_pipeline,
                bam_path,
                input_dir,
                output_dir,
                samtools,
                bam_coverage,
                args,
                selected_case_by_file_id.get(bam_path.parent.name),
                genes,
            ): bam_path
            for bam_path in bams
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="BAM to BigWig to tensor",
            unit="file",
        ):
            bam_path = futures[future]
            try:
                result = future.result()
                pipeline_results.append(result)
                message = f"[OK] {bam_path} -> {result['bigwig']}"
                if result["tensor"] is not None:
                    message += f" -> {result['tensor']['output']}"
                tqdm.write(message)
            except Exception as error:
                failures.append((bam_path, str(error)))
                tqdm.write(f"[ERROR] {bam_path}: {error}")

    print(
        f"[INFO] Completed: {len(bams) - len(failures)}/{len(bams)}"
    )
    if failures:
        print("[WARN] Failed pipeline files:")
        for bam_path, error in failures:
            print(f"  {bam_path}: {error}")

    if not args.skip_tensors:
        tensor_results = [
            result["tensor"]
            for result in pipeline_results
            if result["tensor"] is not None
        ]
        manifest = {
            "genes_file": str(args.genes),
            "sample_sheet": str(args.sample_sheet),
            "bigwig_dir": str(output_dir),
            "output_dir": str(args.tensor_output_dir),
            "bin_bp": args.tensor_bin_bp,
            "chunk_bins": args.chunk_bins,
            "chunk_bp": args.tensor_bin_bp * args.chunk_bins,
            "gene_names": genes["gene"].tolist(),
            "duplicates": duplicates,
            "patients": sorted(
                tensor_results, key=lambda item: item["case_id"]
            ),
            "failures": [
                {"bam": str(bam_path), "error": error}
                for bam_path, error in failures
            ],
        }
        manifest_path = args.tensor_output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(
            f"[INFO] Patient tensors completed: "
            f"{len(tensor_results)}/{len(selected_case_by_file_id)}"
        )
        print(f"[INFO] Tensor manifest: {manifest_path}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
