#!/usr/bin/env python
"""Create POG570 patient-gene AlphaMissense burdens from cohort score files."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm


PROJECT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_MUTATIONS = PROJECT_DIR / "data" / "POG570" / "POG570_small_mutations.txt"
DEFAULT_PATIENT_MAPPING = PROJECT_DIR / "metadata" / "pog570_patient_driver_gene_file_mapping.tsv"
DEFAULT_COHORT_SCORES_DIR = PROJECT_DIR / "data" / "alphagenome_scores" / "pog570_cohort_alphamissense_scores"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data" / "alphagenome_scores" / "pog570_patient_alphamissense_scores_by_cohort"
DEFAULT_COMBINED_OUTPUT = PROJECT_DIR / "data" / "pog570_alphamissense_combined_gene_level_scores.tsv"

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
PATIENT_MUTATION_COLUMNS = [
    "patient_id",
    "analysis_cohort",
    "mutation",
    "gene",
    *SCORE_COLUMNS,
]
COMBINED_COLUMNS = [
    "patient_id",
    "analysis_cohort",
    "gene",
    "combined_pathogenicity_burden",
    "n_scored_mutations",
]


@dataclass
class GeneBurden:
    analysis_cohort: str
    burden: float = 0.0
    n_scored_mutations: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mutations", type=Path, default=DEFAULT_MUTATIONS)
    parser.add_argument("--patient-driver-mapping", type=Path, default=DEFAULT_PATIENT_MAPPING)
    parser.add_argument("--cohort-scores-dir", type=Path, default=DEFAULT_COHORT_SCORES_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--combined-output", type=Path, default=DEFAULT_COMBINED_OUTPUT)
    parser.add_argument("--cohorts", nargs="+", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--combined-only", action="store_true")
    return parser.parse_args()


def mutation_string(chrom: str, pos: str, ref: str, alt: str) -> str | None:
    ref = str(ref).strip().upper()
    alt = str(alt).strip().upper()
    try:
        pos_int = int(pos)
    except (TypeError, ValueError):
        return None
    if len(ref) != 1 or len(alt) != 1 or ref == alt:
        return None
    chrom = str(chrom).strip()
    chrom = chrom[3:] if chrom.lower().startswith("chr") else chrom
    return f"{chrom}:{pos_int}-{pos_int}:{ref}:{alt}"


def pathogenicity_value(row: dict[str, str]) -> float:
    try:
        return float(row["am_pathogenicity"])
    except (KeyError, TypeError, ValueError):
        return float("-inf")


def load_score_lookup(path: Path) -> dict[tuple[str, str], tuple[str, ...]]:
    lookup = {}
    best_scores = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"mutation", "gene", *SCORE_COLUMNS}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required column(s): {sorted(missing)}")
        for row in reader:
            key = row["mutation"], row["gene"]
            score = pathogenicity_value(row)
            if key in best_scores and score <= best_scores[key]:
                continue
            lookup[key] = tuple(row[column] for column in SCORE_COLUMNS)
            best_scores[key] = score
    return lookup


def load_patient_driver_genes(path: Path) -> dict[str, set[str]]:
    mapping = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    patient_genes = {}
    for _, row in mapping.iterrows():
        genes = set()
        for value in row["driver_gene_files"].split(";"):
            if not value.strip():
                continue
            driver_path = Path(value)
            driver_path = driver_path if driver_path.is_absolute() else PROJECT_DIR / driver_path
            genes.update(pd.read_csv(driver_path, sep="\t", dtype=str, usecols=["gene"])["gene"].dropna().astype(str))
        if genes:
            patient_genes[row["patient_id"]] = genes
    return patient_genes


def load_pog570_mutations(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str)
    df = df.dropna(subset=["patient_id", "analysis_cohort", "gene_id", "chrom", "pos", "ref", "alt"]).copy()
    df["mutation"] = [
        mutation_string(chrom, pos, ref, alt)
        for chrom, pos, ref, alt in zip(df["chrom"], df["pos"], df["ref"], df["alt"], strict=False)
    ]
    df = df.dropna(subset=["mutation"])
    return df.rename(columns={"gene_id": "gene"})


def process_cohort(cohort: str, mutations: pd.DataFrame, patient_genes: dict[str, set[str]], scores_dir: Path, output_dir: Path, force: bool) -> Path | None:
    scores_path = scores_dir / f"{cohort}.tsv"
    if not scores_path.exists():
        print(f"Skipping {cohort}: missing cohort score file {scores_path}")
        return None

    output_path = output_dir / f"{cohort}.tsv"
    if output_path.exists() and not force:
        print(f"Skipping completed patient score file: {output_path}")
        return output_path

    score_lookup = load_score_lookup(scores_path)
    cohort_mutations = mutations[mutations["analysis_cohort"].eq(cohort)].copy()
    rows = []
    for row in tqdm(cohort_mutations.itertuples(index=False), total=len(cohort_mutations), desc=cohort, unit="mutation"):
        genes = patient_genes.get(str(row.patient_id), set())
        if row.gene not in genes:
            continue
        score_values = score_lookup.get((row.mutation, row.gene))
        if score_values is None:
            continue
        rows.append([row.patient_id, row.analysis_cohort, row.mutation, row.gene, *score_values])

    output_dir.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(PATIENT_MUTATION_COLUMNS)
        writer.writerows(rows)
    print(f"Wrote {len(rows)} patient-mutation score rows for {cohort}")
    return output_path


def build_combined_gene_scores(paths: list[Path], output_path: Path) -> None:
    burdens: dict[tuple[str, str], GeneBurden] = {}
    for path in paths:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            required = {"patient_id", "analysis_cohort", "gene", "am_pathogenicity"}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{path} is missing required column(s): {sorted(missing)}")
            for row in reader:
                try:
                    score = float(row["am_pathogenicity"])
                except (TypeError, ValueError):
                    continue
                key = row["patient_id"], row["gene"]
                burden = burdens.get(key)
                if burden is None:
                    burden = GeneBurden(analysis_cohort=row["analysis_cohort"])
                    burdens[key] = burden
                burden.burden += score
                burden.n_scored_mutations += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(COMBINED_COLUMNS)
        for (patient_id, gene), burden in sorted(burdens.items()):
            writer.writerow([
                patient_id,
                burden.analysis_cohort,
                gene,
                f"{burden.burden:.12g}",
                burden.n_scored_mutations,
            ])
    print(f"Wrote {len(burdens)} patient-gene scores to {output_path}")


def main() -> None:
    args = parse_args()
    if args.combined_only:
        paths = sorted(args.output_dir.glob("*.tsv"))
        build_combined_gene_scores(paths, args.combined_output)
        return

    mutations = load_pog570_mutations(args.mutations)
    patient_genes = load_patient_driver_genes(args.patient_driver_mapping)
    cohorts = sorted(set(mutations["analysis_cohort"]).intersection(path.stem for path in args.cohort_scores_dir.glob("*.tsv")))
    if args.cohorts:
        cohorts = [cohort for cohort in cohorts if cohort in set(args.cohorts)]

    output_paths = []
    for cohort in cohorts:
        output_path = process_cohort(cohort, mutations, patient_genes, args.cohort_scores_dir, args.output_dir, args.force)
        if output_path is not None:
            output_paths.append(output_path)
    build_combined_gene_scores(output_paths, args.combined_output)


if __name__ == "__main__":
    main()
