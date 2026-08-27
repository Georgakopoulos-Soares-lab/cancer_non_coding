#!/usr/bin/env python
"""Create patient-level AlphaMissense scores from tissue-level score tables.

For each TCGA cancer type, this script:

1. Maps the cancer acronym to its tissue/organ.
2. Loads the corresponding tissue AlphaMissense score table.
3. Streams the cancer's patient variant table.
4. Writes one row per scored patient-gene-mutation.

When a tissue table contains multiple transcript annotations for the same
mutation-gene pair, the row with the highest ``am_pathogenicity`` is retained.
Cancer types are processed in parallel. Final output files are published
atomically and existing files are skipped unless ``--force`` is supplied.
Afterward, the completed per-cancer outputs are aggregated into
``alphamissense_combined_gene_level_scores.tsv`` with one row per patient-gene.
The combined burden is the sum of ``am_pathogenicity`` across all scored
mutations in that patient's gene, without thresholding. The resulting burdens
are also added to patient JSON files under ``gene_alphamissense_burden``.

Usage examples:

    # Process all available TCGA cancer types with the default worker count.
    python scripts/data_preparation/8.1_get_patient_level_alphamissense_score.py

    # Process selected cancer types with eight parallel workers.
    python scripts/data_preparation/8.1_get_patient_level_alphamissense_score.py \
        --cancers GBM BRCA LUAD \
        --workers 8

    # Reprocess cancers even when their final output files already exist.
    python scripts/data_preparation/8.1_get_patient_level_alphamissense_score.py \
        --force

    # Rebuild only the pancancer patient-gene file from existing cancer outputs.
    python scripts/data_preparation/8.1_get_patient_level_alphamissense_score.py \
        --combined-only

    # Use custom input and output locations.
    python scripts/data_preparation/8.1_get_patient_level_alphamissense_score.py \
        --tissue-scores-dir data/alphagenome_scores/tissue_alphamissense_scores \
        --patient-variants-dir data/TCGA/tcga_patient_variants_by_cancer \
        --cancer-mapping metadata/cancer_types_acronyms.tsv \
        --output-dir data/alphagenome_scores/patient_alphamissense_scores_by_cancer
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass
from pathlib import Path

from tqdm.auto import tqdm


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_TISSUE_SCORES_DIR = (
    PROJECT_DIR / "data" / "alphagenome_scores" / "tissue_alphamissense_scores"
)
DEFAULT_PATIENT_VARIANTS_DIR = (
    PROJECT_DIR / "data" / "TCGA" / "tcga_patient_variants_by_cancer"
)
DEFAULT_CANCER_MAPPING = PROJECT_DIR / "metadata" / "cancer_types_acronyms.tsv"
DEFAULT_OUTPUT_DIR = (
    PROJECT_DIR / "data" / "alphagenome_scores" / "patient_alphamissense_scores_by_cancer"
)
DEFAULT_PATIENT_JSON_DIR = PROJECT_DIR / "data" / "patient_json"
COMBINED_OUTPUT_FILE = (
    PROJECT_DIR / "data" / "alphamissense_combined_gene_level_scores.tsv"
)

SCORE_COLUMNS = [
    "CHROM",
    "POS",
    "REF",
    "ALT",
    "genome",
    "uniprot_id",
    "transcript_id",
    "protein_variant",
    "am_pathogenicity",
    "am_class",
]
OUTPUT_COLUMNS = [
    "bcr_patient_barcode",
    "cancer_type",
    "tissue",
    "mutation",
    "gene",
    *SCORE_COLUMNS,
]
COMBINED_OUTPUT_COLUMNS = [
    "bcr_patient_barcode",
    "cancer_type",
    "tissue",
    "gene",
    "combined_pathogenicity_burden",
    "n_scored_mutations",
]


@dataclass(frozen=True)
class CancerJob:
    cancer_type: str
    tissue: str
    variants_path: Path
    scores_path: Path
    output_path: Path


@dataclass(frozen=True)
class CancerResult:
    cancer_type: str
    tissue: str
    input_rows: int
    matched_rows: int
    unique_score_keys: int
    duplicate_score_rows: int
    malformed_rows: int
    input_bytes: int


@dataclass
class GeneBurden:
    cancer_type: str
    tissue: str
    burden: float = 0.0
    n_scored_mutations: int = 0


@dataclass(frozen=True)
class JsonUpdateResult:
    patient_id: str
    changed: bool
    n_genes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join TCGA patient variants to tissue-level AlphaMissense scores."
    )
    parser.add_argument(
        "--tissue-scores-dir",
        type=Path,
        default=DEFAULT_TISSUE_SCORES_DIR,
        help="Directory containing tissue-level AlphaMissense TSVs.",
    )
    parser.add_argument(
        "--patient-variants-dir",
        type=Path,
        default=DEFAULT_PATIENT_VARIANTS_DIR,
        help="Directory containing TCGA *_variants.tsv files.",
    )
    parser.add_argument(
        "--cancer-mapping",
        type=Path,
        default=DEFAULT_CANCER_MAPPING,
        help="TSV mapping cancer acronyms to organs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for one patient-level TSV per cancer type.",
    )
    parser.add_argument(
        "--patient-json-dir",
        type=Path,
        default=DEFAULT_PATIENT_JSON_DIR,
        help="Directory containing patient JSON files to update.",
    )
    parser.add_argument(
        "--cancers",
        nargs="+",
        default=None,
        help="Optional cancer acronyms to process, e.g. GBM BRCA LUAD.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Parallel cancer workers (default: %(default)s).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess cancer types whose final output TSV already exists.",
    )
    parser.add_argument(
        "--combined-only",
        action="store_true",
        help="Skip patient-mutation generation and rebuild only the combined gene-level file.",
    )
    parser.add_argument(
        "--skip-patient-json-update",
        action="store_true",
        help="Generate TSV outputs without updating patient JSON files.",
    )
    return parser.parse_args()


def sanitize_tissue_name(tissue: str) -> str:
    return tissue.strip().replace(" ", "_")


def load_tcga_cancer_tissue_map(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"Acronym", "Organ", "Projects"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required column(s): {sorted(missing)}")

        for row in reader:
            projects = {
                project.strip().upper()
                for project in row["Projects"].split(",")
                if project.strip()
            }
            if "TCGA" not in projects:
                continue
            cancer_type = row["Acronym"].strip()
            tissue = row["Organ"].strip()
            if cancer_type and tissue:
                mapping[cancer_type] = tissue
    return mapping


def pathogenicity_value(row: dict[str, str]) -> float:
    try:
        return float(row["am_pathogenicity"])
    except (KeyError, TypeError, ValueError):
        return float("-inf")


def load_score_lookup(
    path: Path,
) -> tuple[dict[tuple[str, str], tuple[str, ...]], int]:
    """Load one score row per mutation-gene, retaining the highest score."""
    lookup: dict[tuple[str, str], tuple[str, ...]] = {}
    best_scores: dict[tuple[str, str], float] = {}
    duplicate_rows = 0

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"mutation", "gene", *SCORE_COLUMNS}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required column(s): {sorted(missing)}")

        for row in reader:
            mutation = row["mutation"].strip()
            gene = row["gene"].strip()
            if not mutation or not gene:
                continue
            key = mutation, gene
            score = pathogenicity_value(row)
            if key in lookup:
                duplicate_rows += 1
                if score <= best_scores[key]:
                    continue
            lookup[key] = tuple(row[column] for column in SCORE_COLUMNS)
            best_scores[key] = score

    return lookup, duplicate_rows


def process_cancer(job: CancerJob) -> CancerResult:
    """Join one cancer variant file to its tissue scores."""
    temporary_output = job.output_path.with_name(f".{job.output_path.name}.tmp")
    temporary_output.unlink(missing_ok=True)

    score_lookup, duplicate_score_rows = load_score_lookup(job.scores_path)
    input_rows = 0
    matched_rows = 0
    malformed_rows = 0

    try:
        with (
            job.variants_path.open(newline="") as variants_handle,
            temporary_output.open("w", newline="") as output_handle,
        ):
            reader = csv.DictReader(variants_handle, delimiter="\t")
            required = {"bcr_patient_barcode", "mutation", "gene_name"}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"{job.variants_path} is missing required column(s): {sorted(missing)}"
                )

            writer = csv.writer(output_handle, delimiter="\t", lineterminator="\n")
            writer.writerow(OUTPUT_COLUMNS)
            for row in reader:
                input_rows += 1
                patient_id = row["bcr_patient_barcode"].strip()
                mutation = row["mutation"].strip()
                gene = row["gene_name"].strip()
                if not patient_id or not mutation or not gene:
                    malformed_rows += 1
                    continue

                score_values = score_lookup.get((mutation, gene))
                if score_values is None:
                    continue
                writer.writerow(
                    [
                        patient_id,
                        job.cancer_type,
                        job.tissue,
                        mutation,
                        gene,
                        *score_values,
                    ]
                )
                matched_rows += 1

        temporary_output.replace(job.output_path)
    except BaseException:
        temporary_output.unlink(missing_ok=True)
        raise

    return CancerResult(
        cancer_type=job.cancer_type,
        tissue=job.tissue,
        input_rows=input_rows,
        matched_rows=matched_rows,
        unique_score_keys=len(score_lookup),
        duplicate_score_rows=duplicate_score_rows,
        malformed_rows=malformed_rows,
        input_bytes=job.variants_path.stat().st_size,
    )


def build_jobs(
    args: argparse.Namespace,
) -> tuple[list[CancerJob], list[str], list[str]]:
    cancer_tissue_map = load_tcga_cancer_tissue_map(args.cancer_mapping)
    available_variants = {
        path.name.removesuffix("_variants.tsv"): path
        for path in args.patient_variants_dir.glob("*_variants.tsv")
    }

    if args.cancers:
        requested = [cancer.upper() for cancer in args.cancers]
        unknown = [
            cancer
            for cancer in requested
            if cancer not in cancer_tissue_map or cancer not in available_variants
        ]
        if unknown:
            raise ValueError(f"Unknown or unavailable TCGA cancer type(s): {sorted(set(unknown))}")
        cancer_types = requested
    else:
        cancer_types = sorted(set(cancer_tissue_map).intersection(available_variants))

    jobs: list[CancerJob] = []
    completed: list[str] = []
    missing_scores: list[str] = []
    for cancer_type in cancer_types:
        tissue = cancer_tissue_map[cancer_type]
        scores_path = args.tissue_scores_dir / f"{sanitize_tissue_name(tissue)}.tsv"
        output_path = args.output_dir / f"{cancer_type}.tsv"
        (args.output_dir / f".{cancer_type}.tsv.tmp").unlink(missing_ok=True)

        if output_path.is_file() and not args.force:
            completed.append(cancer_type)
            continue
        if not scores_path.is_file():
            missing_scores.append(f"{cancer_type} ({tissue})")
            continue
        jobs.append(
            CancerJob(
                cancer_type=cancer_type,
                tissue=tissue,
                variants_path=available_variants[cancer_type],
                scores_path=scores_path,
                output_path=output_path,
            )
        )
    return jobs, completed, missing_scores


def process_jobs(jobs: list[CancerJob], workers: int) -> list[CancerResult]:
    results: list[CancerResult] = []
    failures: list[str] = []
    total_bytes = sum(job.variants_path.stat().st_size for job in jobs)

    with (
        tqdm(total=len(jobs), desc="Cancer types", unit="cancer", position=0) as cancer_progress,
        tqdm(
            total=total_bytes,
            desc="Variant input processed",
            unit="B",
            unit_scale=True,
            position=1,
        ) as byte_progress,
        ProcessPoolExecutor(max_workers=workers) as executor,
    ):
        futures = {executor.submit(process_cancer, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            byte_progress.update(job.variants_path.stat().st_size)
            cancer_progress.update(1)
            cancer_progress.set_postfix_str(job.cancer_type)
            try:
                results.append(future.result())
            except BaseException as exc:
                failures.append(f"{job.cancer_type}: {exc}")

    if failures:
        failure_text = "\n  ".join(failures)
        raise RuntimeError(f"Failed cancer type(s):\n  {failure_text}")
    return sorted(results, key=lambda result: result.cancer_type)


def completed_cancer_output_paths(
    output_dir: Path,
    cancer_mapping: dict[str, str],
) -> list[Path]:
    return [
        output_dir / f"{cancer_type}.tsv"
        for cancer_type in sorted(cancer_mapping)
        if (output_dir / f"{cancer_type}.tsv").is_file()
    ]


def build_alphamissense_combined_gene_level_scores(
    cancer_output_paths: list[Path],
    output_path: Path,
) -> tuple[int, int, int]:
    """Sum all mutation pathogenicity scores into patient-gene burdens."""
    if not cancer_output_paths:
        raise FileNotFoundError(
            f"No completed per-cancer patient score TSVs found in {output_path.parent}"
        )

    burdens: dict[tuple[str, str], GeneBurden] = {}
    input_rows = 0
    invalid_score_rows = 0
    total_bytes = sum(path.stat().st_size for path in cancer_output_paths)

    with tqdm(
        total=total_bytes,
        desc="Combining patient-gene scores",
        unit="B",
        unit_scale=True,
    ) as progress:
        for path in cancer_output_paths:
            with path.open(newline="") as handle:
                header_line = handle.readline()
                progress.update(len(header_line))
                if not header_line:
                    raise ValueError(f"Empty patient score file: {path}")
                fieldnames = header_line.rstrip("\r\n").split("\t")
                required = {
                    "bcr_patient_barcode",
                    "cancer_type",
                    "tissue",
                    "mutation",
                    "gene",
                    "am_pathogenicity",
                }
                missing = required.difference(fieldnames)
                if missing:
                    raise ValueError(f"{path} is missing required column(s): {sorted(missing)}")

                reader = csv.DictReader(handle, fieldnames=fieldnames, delimiter="\t")
                for row in reader:
                    progress.update(sum(len(value) for value in row.values()) + len(row))
                    input_rows += 1
                    patient_id = row["bcr_patient_barcode"].strip()
                    cancer_type = row["cancer_type"].strip()
                    tissue = row["tissue"].strip()
                    gene = row["gene"].strip()
                    if not patient_id or not cancer_type or not tissue or not gene:
                        invalid_score_rows += 1
                        continue
                    try:
                        score = float(row["am_pathogenicity"])
                    except (TypeError, ValueError):
                        invalid_score_rows += 1
                        continue

                    key = patient_id, gene
                    burden = burdens.get(key)
                    if burden is None:
                        burden = GeneBurden(cancer_type=cancer_type, tissue=tissue)
                        burdens[key] = burden
                    elif burden.cancer_type != cancer_type or burden.tissue != tissue:
                        raise ValueError(
                            f"Patient {patient_id} appears under conflicting cancer/tissue labels."
                        )

                    burden.n_scored_mutations += 1
                    burden.burden += score

    temporary_output = output_path.with_name(f".{output_path.name}.tmp")
    temporary_output.unlink(missing_ok=True)
    try:
        with temporary_output.open("w", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(COMBINED_OUTPUT_COLUMNS)
            for (patient_id, gene), burden in tqdm(
                sorted(burdens.items()),
                desc="Writing combined scores",
                unit="patient-gene",
            ):
                writer.writerow(
                    [
                        patient_id,
                        burden.cancer_type,
                        burden.tissue,
                        gene,
                        f"{burden.burden:.12g}",
                        burden.n_scored_mutations,
                    ]
                )
        temporary_output.replace(output_path)
    except BaseException:
        temporary_output.unlink(missing_ok=True)
        raise

    return len(burdens), input_rows, invalid_score_rows


def iter_patient_gene_burdens(
    combined_path: Path,
):
    """Yield sorted patient IDs and their gene burdens from the combined TSV."""
    with combined_path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {
            "bcr_patient_barcode",
            "gene",
            "combined_pathogenicity_burden",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{combined_path} is missing required column(s): {sorted(missing)}")

        current_patient: str | None = None
        burdens: dict[str, float] = {}
        for row in reader:
            patient_id = row["bcr_patient_barcode"].strip()
            gene = row["gene"].strip()
            if not patient_id or not gene:
                continue
            try:
                burden = float(row["combined_pathogenicity_burden"])
            except (TypeError, ValueError):
                raise ValueError(
                    f"Invalid combined burden for patient {patient_id}, gene {gene}."
                )

            if current_patient is not None and patient_id < current_patient:
                raise ValueError(f"{combined_path} is not sorted by patient ID.")
            if current_patient is not None and patient_id != current_patient:
                yield current_patient, burdens
                burdens = {}
            current_patient = patient_id
            burdens[gene] = burden

        if current_patient is not None:
            yield current_patient, burdens


def update_patient_json(
    json_path: Path,
    gene_burdens: dict[str, float],
) -> JsonUpdateResult:
    """Atomically add or replace gene_alphamissense_burden in one JSON."""
    patient_id = json_path.stem
    with json_path.open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Patient JSON must contain an object: {json_path}")

    sorted_burdens = dict(sorted(gene_burdens.items()))
    if payload.get("gene_alphamissense_burden") == sorted_burdens:
        return JsonUpdateResult(patient_id=patient_id, changed=False, n_genes=len(sorted_burdens))

    payload["gene_alphamissense_burden"] = sorted_burdens
    temporary_path = json_path.with_name(f".{json_path.name}.tmp")
    temporary_path.unlink(missing_ok=True)
    try:
        with temporary_path.open("w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary_path.replace(json_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    return JsonUpdateResult(patient_id=patient_id, changed=True, n_genes=len(sorted_burdens))


def update_patient_json_files(
    combined_path: Path,
    patient_json_dir: Path,
    workers: int,
) -> tuple[int, int, int, int]:
    """Merge sorted burdens into patient JSON files with bounded parallelism."""
    json_paths = sorted(patient_json_dir.glob("*.json"), key=lambda path: path.stem)
    if not json_paths:
        raise FileNotFoundError(f"No patient JSON files found in {patient_json_dir}")

    burden_iter = iter(iter_patient_gene_burdens(combined_path))
    current_burden = next(burden_iter, None)
    missing_json_patients = 0
    updated = 0
    unchanged = 0
    total_genes = 0
    max_pending = max(workers * 4, 1)

    def collect_completed(done) -> None:
        nonlocal updated, unchanged, total_genes
        for future in done:
            result = future.result()
            updated += int(result.changed)
            unchanged += int(not result.changed)
            total_genes += result.n_genes

    pending = set()
    with (
        ThreadPoolExecutor(max_workers=workers) as executor,
        tqdm(total=len(json_paths), desc="Updating patient JSON", unit="patient") as progress,
    ):
        for json_path in json_paths:
            patient_id = json_path.stem
            while current_burden is not None and current_burden[0] < patient_id:
                missing_json_patients += 1
                current_burden = next(burden_iter, None)

            if current_burden is not None and current_burden[0] == patient_id:
                gene_burdens = current_burden[1]
                current_burden = next(burden_iter, None)
            else:
                gene_burdens = {}

            pending.add(executor.submit(update_patient_json, json_path, gene_burdens))
            if len(pending) >= max_pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                collect_completed(done)
                progress.update(len(done))

        while current_burden is not None:
            missing_json_patients += 1
            current_burden = next(burden_iter, None)

        for future in as_completed(pending):
            collect_completed([future])
            progress.update(1)

    return updated, unchanged, total_genes, missing_json_patients


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        sys.exit("--workers must be positive.")
    if not args.cancer_mapping.is_file():
        sys.exit(f"Cancer mapping file not found: {args.cancer_mapping}")
    if not args.skip_patient_json_update and not args.patient_json_dir.is_dir():
        sys.exit(f"Patient JSON directory not found: {args.patient_json_dir}")
    if not args.combined_only:
        for path, description in [
            (args.tissue_scores_dir, "Tissue score directory"),
            (args.patient_variants_dir, "Patient variant directory"),
        ]:
            if not path.is_dir():
                sys.exit(f"{description} not found: {path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[CancerResult] = []
    json_update_stats: tuple[int, int, int, int] | None = None
    try:
        if not args.combined_only:
            jobs, completed, missing_scores = build_jobs(args)
            if completed:
                print(f"Skipping {len(completed)} completed cancer type(s): {', '.join(completed)}")
            if missing_scores:
                print(
                    "Skipping cancer type(s) without tissue score files: "
                    + ", ".join(missing_scores),
                    file=sys.stderr,
                )
            if jobs:
                print(f"Processing {len(jobs)} cancer type(s) with {args.workers} worker(s).")
                results = process_jobs(jobs, args.workers)
            else:
                print("No cancer types require patient-mutation processing.")

        cancer_mapping = load_tcga_cancer_tissue_map(args.cancer_mapping)
        cancer_output_paths = completed_cancer_output_paths(args.output_dir, cancer_mapping)
        combined_output_path = COMBINED_OUTPUT_FILE
        n_patient_genes, n_score_rows, n_invalid_rows = build_alphamissense_combined_gene_level_scores(
            cancer_output_paths,
            combined_output_path,
        )
        if not args.skip_patient_json_update:
            json_update_stats = update_patient_json_files(
                combined_output_path,
                args.patient_json_dir,
                args.workers,
            )
    except (OSError, RuntimeError, ValueError) as exc:
        sys.exit(f"Error: {exc}")

    if results:
        print("\nPer-cancer summary:")
        for result in results:
            print(
                f"  {result.cancer_type} ({result.tissue}): "
                f"{result.matched_rows:,}/{result.input_rows:,} patient variant rows matched; "
                f"{result.unique_score_keys:,} score keys; "
                f"{result.duplicate_score_rows:,} lower-scoring transcript rows collapsed"
            )
    print(
        f"\nCombined gene-level scores: {n_patient_genes:,} patient-gene rows "
        f"from {n_score_rows:,} mutation score rows "
        f"({n_invalid_rows:,} invalid rows skipped)."
    )
    print(f"Combined output: {combined_output_path}")
    if json_update_stats is not None:
        updated, unchanged, total_genes, missing_json_patients = json_update_stats
        print(
            f"Patient JSON updates: {updated:,} changed, {unchanged:,} already current; "
            f"{total_genes:,} gene burdens assigned."
        )
        if missing_json_patients:
            print(
                f"Warning: {missing_json_patients:,} scored patient(s) had no JSON file.",
                file=sys.stderr,
            )
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
