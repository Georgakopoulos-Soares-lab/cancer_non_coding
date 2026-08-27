#!/usr/bin/env python
"""Create POG570 patient-gene hotspot status files from hg19 variants."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_MUTATION_DIR = PROJECT_DIR / "data" / "PCAWG_POG570_mutations"
DEFAULT_PATIENT_LIST = PROJECT_DIR / "metadata" / "POG570_patient_list.txt"
DEFAULT_HOTSPOTS = PROJECT_DIR / "data" / "hotspots_v2" / "driver_mutation_organ_mapping.long.tsv"
DEFAULT_PANCANCER_GENES = PROJECT_DIR / "data" / "driver_genes_coords" / "Pancancer_1pc.tsv"
DEFAULT_PATIENT_DRIVER_MAPPING = PROJECT_DIR / "metadata" / "pog570_patient_driver_gene_file_mapping.tsv"
DEFAULT_COHORT_MAPPING = PROJECT_DIR / "metadata" / "POG570_cohorts.tsv"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data" / "pog570_hotspot_gene_status"

DNA_BASES = set("ACGT")
OUTPUT_COLUMNS = ["patient", "gene", "hotspot_organ", "has_hotspot"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mutation-dir", type=Path, default=DEFAULT_MUTATION_DIR)
    parser.add_argument("--patient-list", type=Path, default=DEFAULT_PATIENT_LIST)
    parser.add_argument("--hotspots", type=Path, default=DEFAULT_HOTSPOTS)
    parser.add_argument("--pancancer-genes", type=Path, default=DEFAULT_PANCANCER_GENES)
    parser.add_argument("--patient-driver-mapping", type=Path, default=DEFAULT_PATIENT_DRIVER_MAPPING)
    parser.add_argument("--cohort-mapping", type=Path, default=DEFAULT_COHORT_MAPPING)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def normalize_chrom(chrom: object) -> str:
    chrom = str(chrom).strip()
    return chrom[3:] if chrom.lower().startswith("chr") else chrom


def load_patient_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def load_genes(path: Path) -> list[str]:
    genes = pd.read_csv(path, sep="\t", dtype=str, usecols=["gene"])["gene"].dropna()
    return sorted(set(genes.astype(str)))


def load_driver_file_genes(path: Path) -> set[str]:
    path = path if path.is_absolute() else PROJECT_DIR / path
    return set(pd.read_csv(path, sep="\t", dtype=str, usecols=["gene"])["gene"].dropna().astype(str))


def load_patient_driver_genes(path: Path) -> tuple[dict[str, set[str]], dict[str, str], dict[str, list[str]]]:
    mapping = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    driver_file_cache: dict[str, set[str]] = {}
    patient_genes: dict[str, set[str]] = {}
    patient_cohort: dict[str, str] = {}
    cohort_patients: dict[str, list[str]] = defaultdict(list)

    for _, row in mapping.iterrows():
        patient = row["patient_id"]
        cohort = row["analysis_cohort"]
        patient_cohort[patient] = cohort
        genes: set[str] = set()
        for driver_file in row["driver_gene_files"].split(";"):
            if not driver_file:
                continue
            if driver_file not in driver_file_cache:
                driver_file_cache[driver_file] = load_driver_file_genes(Path(driver_file))
            genes.update(driver_file_cache[driver_file])
        if genes:
            patient_genes[patient] = genes
            cohort_patients[cohort].append(patient)

    return patient_genes, patient_cohort, dict(cohort_patients)


def load_cohort_hotspot_organs(path: Path) -> dict[str, set[str]]:
    mapping = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    cohort_organs: dict[str, set[str]] = {}
    for _, row in mapping.iterrows():
        organs = {organ.strip() for organ in row["hotspot_organ"].split(",") if organ.strip()}
        cohort_organs[row["cohort"]] = organs
    return cohort_organs


def parse_mutation(mutation: str) -> tuple[str, int, int, str, str] | None:
    parts = str(mutation).split(":")
    if len(parts) != 4:
        return None
    chrom, coord, ref, alt = parts
    try:
        if "-" in coord:
            start, end = map(int, coord.split("-", 1))
        else:
            start = end = int(coord)
    except ValueError:
        return None
    return normalize_chrom(chrom), min(start, end), max(start, end), ref.upper(), alt.upper()


def is_snv(ref: str, alt: str) -> bool:
    return len(ref) == 1 and len(alt) == 1 and ref in DNA_BASES and alt in DNA_BASES and ref != alt


def load_hotspot_keys(
    path: Path,
    allowed_genes: set[str],
    allowed_organs: set[str],
) -> tuple[set[tuple[str, str, str, int, str, str]], dict[tuple[str, str, str], set[int]]]:
    columns = [
        "gene",
        "hg19_chromosome",
        "hg19_start",
        "hg19_end",
        "hotspot_organ",
        "parseable_snv_alleles",
        "source_type",
    ]
    hotspots = pd.read_csv(path, sep="\t", dtype=str, usecols=columns).dropna(
        subset=["gene", "hg19_chromosome", "hg19_start", "hotspot_organ", "source_type"]
    )
    hotspots = hotspots[hotspots["gene"].isin(allowed_genes) & hotspots["hotspot_organ"].isin(allowed_organs)]

    snv_keys: set[tuple[str, str, str, int, str, str]] = set()
    indel_sites: dict[tuple[str, str, str], set[int]] = defaultdict(set)

    for row in hotspots.itertuples(index=False):
        gene = row.gene
        hotspot_organ = row.hotspot_organ
        chrom = normalize_chrom(row.hg19_chromosome)
        start = int(row.hg19_start)
        end = int(row.hg19_end) if pd.notna(row.hg19_end) else start

        if row.source_type == "SNV":
            for allele in str(row.parseable_snv_alleles).split("|"):
                if ">" not in allele:
                    continue
                ref, alt = allele.split(">", 1)
                if is_snv(ref, alt):
                    snv_keys.add((hotspot_organ, gene, chrom, start, ref, alt))
        else:
            for pos in range(min(start, end), max(start, end) + 1):
                indel_sites[(hotspot_organ, gene, chrom)].add(pos)

    return snv_keys, dict(indel_sites)


def patient_hotspot_genes(
    patient: str,
    mutation_dir: Path,
    allowed_genes: set[str],
    hotspot_organs: set[str],
    snv_keys: set[tuple[str, str, str, int, str, str]],
    indel_sites: dict[tuple[str, str, str], set[int]],
) -> set[str]:
    path = mutation_dir / f"{patient}.tsv"
    if not path.exists() or not hotspot_organs:
        return set()

    hits: set[str] = set()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            gene = row.get("gene", "")
            if gene not in allowed_genes:
                continue
            parsed = parse_mutation(row.get("mutation", ""))
            if parsed is None:
                continue
            chrom, start, end, ref, alt = parsed
            for hotspot_organ in hotspot_organs:
                if is_snv(ref, alt) and (hotspot_organ, gene, chrom, start, ref, alt) in snv_keys:
                    hits.add(gene)
                    break
                if any(pos in indel_sites.get((hotspot_organ, gene, chrom), set()) for pos in range(start, end + 1)):
                    hits.add(gene)
                    break

    return hits


def write_matrix(
    path: Path,
    patients: list[str],
    genes_by_patient: dict[str, list[str]],
    hits_by_patient: dict[str, set[str]],
    hotspot_organs_by_patient: dict[str, set[str]],
) -> None:
    rows = []
    for patient in patients:
        hotspot_organ = ",".join(sorted(hotspot_organs_by_patient.get(patient, set())))
        for gene in genes_by_patient.get(patient, []):
            rows.append((patient, gene, hotspot_organ, gene in hits_by_patient.get(patient, set())))
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=OUTPUT_COLUMNS).to_csv(path, sep="\t", index=False)


def main() -> None:
    args = parse_args()

    patients = load_patient_ids(args.patient_list)
    pancancer_genes = load_genes(args.pancancer_genes)
    patient_driver_genes, patient_cohort, cohort_patients = load_patient_driver_genes(args.patient_driver_mapping)
    cohort_hotspot_organs = load_cohort_hotspot_organs(args.cohort_mapping)
    hotspot_organs_by_patient = {
        patient: cohort_hotspot_organs.get(patient_cohort.get(patient, ""), set())
        for patient in patients
    }
    all_genes = set(pancancer_genes).union(*(patient_driver_genes.values() or [set()]))
    all_hotspot_organs = set().union(*(hotspot_organs_by_patient.values() or [set()]))

    snv_keys, indel_sites = load_hotspot_keys(args.hotspots, all_genes, all_hotspot_organs)
    hits_by_patient = {
        patient: patient_hotspot_genes(
            patient,
            args.mutation_dir,
            all_genes,
            hotspot_organs_by_patient.get(patient, set()),
            snv_keys,
            indel_sites,
        )
        for patient in patients
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pancancer_path = args.output_dir / "pog570_pancancer_driver_gene_hotspots.tsv"
    write_matrix(
        pancancer_path,
        patients,
        {patient: pancancer_genes for patient in patients},
        hits_by_patient,
        hotspot_organs_by_patient,
    )
    print(f"Wrote {pancancer_path}")

    cohort_dir = args.output_dir / "cohort_specific"
    for cohort, cohort_patient_ids in sorted(cohort_patients.items()):
        cohort_patient_ids = [patient for patient in cohort_patient_ids if patient in patients]
        genes_by_patient = {
            patient: sorted(patient_driver_genes[patient])
            for patient in cohort_patient_ids
            if patient in patient_driver_genes
        }
        output_path = cohort_dir / f"{cohort}.tsv"
        write_matrix(output_path, cohort_patient_ids, genes_by_patient, hits_by_patient, hotspot_organs_by_patient)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
