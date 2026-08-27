#!/usr/bin/env python
"""Match POG570 cohort driver SNVs to AlphaMissense hg38 scores."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from tqdm.auto import tqdm


PROJECT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_VARIANTS_DIR = PROJECT_DIR / "data" / "alphagenome_scores" / "pog570_cohort_unique_driver_mutations"
DEFAULT_ALPHAMISSENSE = PROJECT_DIR / "data" / "AlphaMissense_hg38.tsv"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data" / "alphagenome_scores" / "pog570_cohort_alphamissense_scores"

AM_COLUMNS = [
    "genome",
    "uniprot_id",
    "transcript_id",
    "protein_variant",
    "am_pathogenicity",
    "am_class",
]
OUTPUT_COLUMNS = [
    "analysis_cohort",
    "mutation",
    "gene",
    "CHROM",
    "POS",
    "REF",
    "ALT",
    *AM_COLUMNS,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants-dir", type=Path, default=DEFAULT_VARIANTS_DIR)
    parser.add_argument("--alphamissense", type=Path, default=DEFAULT_ALPHAMISSENSE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cohorts", nargs="+", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-alphamissense-rows", type=int, default=None)
    return parser.parse_args()


def normalized_chrom(chrom: str) -> str:
    chrom = str(chrom).strip()
    return chrom[3:] if chrom.lower().startswith("chr") else chrom


def read_alphamissense_header(handle) -> list[str]:
    for line in handle:
        if line.startswith("#CHROM\t"):
            return line[1:].rstrip("\r\n").split("\t")
        if not line.startswith("#"):
            return line.rstrip("\r\n").split("\t")
    raise ValueError("AlphaMissense header was not found.")


def load_targets(variants_dir: Path, cohorts: list[str] | None, output_dir: Path, force: bool):
    targets = defaultdict(list)
    variant_paths = sorted(variants_dir.glob("*.tsv"))
    if cohorts:
        requested = set(cohorts)
        variant_paths = [path for path in variant_paths if path.stem in requested]
    for path in variant_paths:
        output_path = output_dir / path.name
        if output_path.exists() and not force:
            print(f"Skipping completed cohort score file: {output_path}")
            continue
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            required = {"analysis_cohort", "mutation", "gene", "chrom", "pos", "ref", "alt"}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{path} is missing required column(s): {sorted(missing)}")
            for row in reader:
                key = (
                    normalized_chrom(row["chrom"]),
                    str(int(row["pos"])),
                    row["ref"].upper(),
                    row["alt"].upper(),
                )
                targets[key].append((row["analysis_cohort"], row["mutation"], row["gene"]))
    return targets


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    targets = load_targets(args.variants_dir, args.cohorts, args.output_dir, args.force)
    if not targets:
        print("No cohort variants to score.")
        return

    handles = {}
    writers = {}
    matched_rows = 0
    try:
        with args.alphamissense.open(newline="") as handle:
            header = read_alphamissense_header(handle)
            required = {"CHROM", "POS", "REF", "ALT", *AM_COLUMNS}
            missing = required.difference(header)
            if missing:
                raise ValueError(f"{args.alphamissense} is missing required column(s): {sorted(missing)}")

            reader = csv.DictReader(handle, fieldnames=header, delimiter="\t")
            for row_index, row in enumerate(
                tqdm(reader, total=args.max_alphamissense_rows, desc="Scanning AlphaMissense", unit="row"),
                start=1,
            ):
                if args.max_alphamissense_rows and row_index > args.max_alphamissense_rows:
                    break
                key = (
                    normalized_chrom(row["CHROM"]),
                    row["POS"],
                    row["REF"].upper(),
                    row["ALT"].upper(),
                )
                target_rows = targets.get(key)
                if not target_rows:
                    continue
                score_values = [row[column] for column in AM_COLUMNS]
                for cohort, mutation, gene in target_rows:
                    if cohort not in writers:
                        output_path = args.output_dir / f"{cohort}.tsv"
                        handles[cohort] = output_path.open("w", newline="")
                        writers[cohort] = csv.writer(handles[cohort], delimiter="\t", lineterminator="\n")
                        writers[cohort].writerow(OUTPUT_COLUMNS)
                    writers[cohort].writerow(
                        [
                            cohort,
                            mutation,
                            gene,
                            row["CHROM"],
                            row["POS"],
                            row["REF"],
                            row["ALT"],
                            *score_values,
                        ]
                    )
                    matched_rows += 1
    finally:
        for handle in handles.values():
            handle.close()

    print(f"Wrote {matched_rows:,} scored cohort-mutation rows to {args.output_dir}")


if __name__ == "__main__":
    main()
