#!/usr/bin/env python
"""Create cohort-level unique POG570 driver-gene SNV files for AlphaMissense."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_MUTATION_DIR = PROJECT_DIR / "data" / "PCAWG_POG570_mutations_hg38"
DEFAULT_MAPPING = PROJECT_DIR / "metadata" / "pog570_cohort_driver_gene_file_mapping.tsv"
DEFAULT_PATIENT_MAPPING = PROJECT_DIR / "metadata" / "pog570_patient_driver_gene_file_mapping.tsv"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data" / "alphagenome_scores" / "pog570_cohort_unique_driver_mutations"

DNA_BASES = frozenset("ACGT")
OUTPUT_COLUMNS = [
    "analysis_cohort",
    "mutation",
    "gene",
    "chrom",
    "pos",
    "ref",
    "alt",
    "n_patients",
    "n_mutation_rows",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mutation-dir", type=Path, default=DEFAULT_MUTATION_DIR)
    parser.add_argument("--cohort-driver-mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--patient-driver-mapping", type=Path, default=DEFAULT_PATIENT_MAPPING)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cohorts", nargs="+", default=None)
    return parser.parse_args()


def normalized_chrom(chrom: str) -> str:
    chrom = str(chrom).strip()
    return chrom[3:] if chrom.lower().startswith("chr") else chrom


def mutation_string(chrom: str, pos: str, ref: str, alt: str) -> str | None:
    chrom = normalized_chrom(chrom)
    ref = str(ref).strip().upper()
    alt = str(alt).strip().upper()
    try:
        pos_int = int(pos)
    except (TypeError, ValueError):
        return None
    if ref not in DNA_BASES or alt not in DNA_BASES or ref == alt:
        return None
    return f"{chrom}:{pos_int}-{pos_int}:{ref}:{alt}"


def parse_mutation(mutation: str) -> tuple[str, str, str, str, str] | None:
    parts = str(mutation).strip().split(":")
    if len(parts) != 4:
        return None
    chrom, span, ref, alt = parts
    if "-" not in span:
        return None
    start, end = span.split("-", 1)
    if start != end:
        return None
    normalized = mutation_string(chrom, start, ref, alt)
    if normalized is None:
        return None
    return normalized, normalized_chrom(chrom), str(int(start)), ref.upper(), alt.upper()


def load_cohort_driver_genes(mapping_path: Path) -> dict[str, set[str]]:
    mapping = pd.read_csv(mapping_path, sep="\t", dtype=str).fillna("")
    cohort_genes = {}
    for _, row in mapping.iterrows():
        cohort = row["analysis_cohort"]
        genes = set()
        for value in row["driver_gene_files"].split(";"):
            if not value.strip():
                continue
            path = Path(value)
            path = path if path.is_absolute() else PROJECT_DIR / path
            driver_df = pd.read_csv(path, sep="\t", dtype=str, usecols=["gene"])
            genes.update(driver_df["gene"].dropna().astype(str))
        if genes:
            cohort_genes[cohort] = genes
    return cohort_genes


def load_patient_cohorts(mapping_path: Path) -> dict[str, str]:
    mapping = pd.read_csv(mapping_path, sep="\t", dtype=str)
    return dict(zip(mapping["patient_id"].astype(str), mapping["analysis_cohort"].astype(str), strict=False))


def load_hg38_driver_mutations(
    mutation_dir: Path,
    patient_cohorts: dict[str, str],
    cohort_genes: dict[str, set[str]],
) -> pd.DataFrame:
    rows = []
    for patient_id, cohort in sorted(patient_cohorts.items()):
        genes = cohort_genes.get(cohort)
        if not genes:
            continue
        path = mutation_dir / f"{patient_id}.tsv"
        if not path.exists():
            continue
        df = pd.read_csv(path, sep="\t", dtype=str)
        if not {"mutation", "gene"}.issubset(df.columns):
            continue
        df = df.dropna(subset=["mutation", "gene"])
        df = df[df["gene"].isin(genes)]
        for mutation, gene in zip(df["mutation"], df["gene"], strict=False):
            parsed = parse_mutation(mutation)
            if parsed is None:
                continue
            normalized, chrom, pos, ref, alt = parsed
            rows.append({
                "analysis_cohort": cohort,
                "patient_id": patient_id,
                "mutation": normalized,
                "gene": gene,
                "chrom": chrom,
                "pos": pos,
                "ref": ref,
                "alt": alt,
            })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    cohort_genes = load_cohort_driver_genes(args.cohort_driver_mapping)
    if args.cohorts:
        requested = set(args.cohorts)
        cohort_genes = {cohort: genes for cohort, genes in cohort_genes.items() if cohort in requested}

    patient_cohorts = load_patient_cohorts(args.patient_driver_mapping)
    mutations = load_hg38_driver_mutations(args.mutation_dir, patient_cohorts, cohort_genes)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for cohort, genes in sorted(cohort_genes.items()):
        output_path = args.output_dir / f"{cohort}.tsv"
        output_path.unlink(missing_ok=True)
        cohort_df = mutations[mutations["analysis_cohort"].eq(cohort)].copy()
        if cohort_df.empty:
            print(f"No hg38 driver SNVs for {cohort}")
            continue

        rows = (
            cohort_df.groupby(["analysis_cohort", "mutation", "gene", "chrom", "pos", "ref", "alt"], as_index=False)
            .agg(
                n_patients=("patient_id", "nunique"),
                n_mutation_rows=("patient_id", "size"),
            )
            .sort_values(["gene", "mutation"])
        )
        rows[OUTPUT_COLUMNS].to_csv(output_path, sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)
        print(f"Wrote {len(rows)} unique driver SNVs for {cohort}")


if __name__ == "__main__":
    main()
