#!/usr/bin/env python
"""Create TCGA cancer variant files limited to driver genes and annotate hotspots."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_VARIANT_DIR = PROJECT_DIR / "data" / "TCGA" / "tcga_patient_variants_by_cancer"
DEFAULT_DRIVER_DIR = PROJECT_DIR / "data" / "driver_genes_coords"
DEFAULT_HOTSPOTS = PROJECT_DIR / "data" / "hotspots_v2" / "hotspots_tcga.tsv"
DEFAULT_HOTSPOTS_HG38_BED = PROJECT_DIR / "data" / "hotspots_v2" / "hotspots_tcga_hg38.bed"
DEFAULT_CANCER_ORGAN_MAP = PROJECT_DIR / "metadata" / "cancer_types_acronyms.tsv"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data" / "TCGA" / "tcga_driver_gene_hotspot_variants_by_cancer"
MAX_DRIVER_GENES = 30
CHUNK_SIZE = 500_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant-dir", type=Path, default=DEFAULT_VARIANT_DIR)
    parser.add_argument("--driver-dir", type=Path, default=DEFAULT_DRIVER_DIR)
    parser.add_argument("--hotspots", type=Path, default=DEFAULT_HOTSPOTS)
    parser.add_argument("--hotspots-hg38-bed", type=Path, default=DEFAULT_HOTSPOTS_HG38_BED)
    parser.add_argument("--cancer-organ-map", type=Path, default=DEFAULT_CANCER_ORGAN_MAP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-driver-genes", type=int, default=MAX_DRIVER_GENES)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    return parser.parse_args()


def normalize_chrom(chrom: object) -> str:
    chrom = str(chrom).strip()
    return chrom[3:] if chrom.lower().startswith("chr") else chrom


def parse_mutation(mutation: object) -> tuple[str, int, int] | None:
    parts = str(mutation).split(":")
    if len(parts) != 4:
        return None
    chrom, coords, _, _ = parts
    try:
        if "-" in coords:
            start, end = map(int, coords.split("-", 1))
        else:
            start = end = int(coords)
    except ValueError:
        return None
    return normalize_chrom(chrom), min(start, end), max(start, end)


def load_driver_genes(path: Path, max_genes: int) -> list[str]:
    genes = pd.read_csv(path, sep="\t", dtype=str, usecols=["gene"])["gene"].dropna()
    return list(dict.fromkeys(genes.astype(str)))[:max_genes]


def load_cancer_hotspot_cohorts(path: Path) -> dict[str, set[str]]:
    mapping = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    mapping = mapping[mapping["Projects"].str.contains("TCGA")]
    cancer_cohorts: dict[str, set[str]] = {}
    for _, row in mapping.iterrows():
        cohorts = {cohort.strip() for cohort in row["Hotspot cohort"].split(",") if cohort.strip()}
        if cohorts:
            cancer_cohorts.setdefault(row["Acronym"], set()).update(cohorts)
    return cancer_cohorts


def load_hg38_position_map(path: Path) -> dict[str, str]:
    bed = pd.read_csv(path, sep="\t", header=None, dtype=str)
    hg19 = bed[3].str.split("-").str[0].str.replace("chr", "", regex=False)
    hg38 = bed[0].str.replace("chr", "", regex=False) + ":" + bed[2].astype(str)
    return dict(zip(hg19, hg38))


def load_hotspots(path: Path, hg38_bed: Path) -> set[tuple[str, str]]:
    hotspots = pd.read_csv(path, sep="\t", dtype=str)
    hg38_position_by_hg19 = load_hg38_position_map(hg38_bed)
    hotspots["Genomic_Position"] = hotspots["Genomic_Position"].map(hg38_position_by_hg19)
    hotspots = hotspots.dropna(subset=["Genomic_Position", "Organ_Types"])
    return set(zip(hotspots["Organ_Types"], hotspots["Genomic_Position"]))


def matching_hotspot_sources(
    mutation: object,
    hotspot_cohorts: set[str],
    hotspots: set[tuple[str, str]],
) -> str:
    parsed = parse_mutation(mutation)
    if parsed is None:
        return ""
    chrom, start, end = parsed
    mutation_positions = {f"{chrom}:{pos}" for pos in range(start, end + 1)}
    matched_cohorts = {
        cohort
        for cohort in hotspot_cohorts
        for mutation_pos in mutation_positions
        if (cohort, mutation_pos) in hotspots
    }
    return ";".join(sorted(matched_cohorts))


def annotate_chunk(chunk: pd.DataFrame, hotspot_cohorts: set[str], hotspots: set[tuple[str, str]]) -> pd.DataFrame:
    chunk = chunk.copy()
    chunk["hotspot_organ"] = ",".join(sorted(hotspot_cohorts))
    chunk["hotspot_source"] = [
        matching_hotspot_sources(mutation, hotspot_cohorts, hotspots)
        for mutation in chunk["mutation"]
    ]
    chunk["is_hotspot"] = chunk["hotspot_source"] != ""
    return chunk


def process_cancer_file(
    variant_path: Path,
    driver_path: Path,
    output_path: Path,
    hotspot_cohorts: set[str],
    hotspots: set[tuple[str, str]],
    max_driver_genes: int,
    chunk_size: int,
) -> tuple[str, int, int, int, int]:
    cancer_type = variant_path.name.removesuffix("_variants.tsv")
    driver_genes = load_driver_genes(driver_path, max_driver_genes)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    n_variants = 0
    n_hotspots = 0
    observed_genes = set()
    wrote_header = False
    for chunk in pd.read_csv(variant_path, sep="\t", dtype=str, chunksize=chunk_size):
        chunk = chunk[chunk["gene_name"].isin(driver_genes)]
        if chunk.empty:
            continue
        chunk = annotate_chunk(chunk, hotspot_cohorts, hotspots)
        chunk.to_csv(output_path, sep="\t", index=False, mode="a", header=not wrote_header)
        wrote_header = True
        n_variants += len(chunk)
        n_hotspots += int(chunk["is_hotspot"].sum())
        observed_genes.update(chunk["gene_name"].dropna())

    if not wrote_header:
        pd.DataFrame(
            columns=["bcr_patient_barcode", "mutation", "gene_name", "hotspot_organ", "hotspot_source", "is_hotspot"]
        ).to_csv(
            output_path,
            sep="\t",
            index=False,
        )
    return cancer_type, len(driver_genes), n_variants, n_hotspots, len(observed_genes)


def main() -> None:
    args = parse_args()
    hotspot_cohorts_by_cancer = load_cancer_hotspot_cohorts(args.cancer_organ_map)
    hotspots = load_hotspots(args.hotspots, args.hotspots_hg38_bed)
    results = []

    for variant_path in sorted(args.variant_dir.glob("*_variants.tsv")):
        cancer_type = variant_path.name.removesuffix("_variants.tsv")
        driver_path = args.driver_dir / f"{cancer_type}.tsv"
        if not driver_path.exists():
            print(f"Skipping {cancer_type}: missing {driver_path}")
            continue
        hotspot_cohorts = hotspot_cohorts_by_cancer.get(cancer_type, set())
        if not hotspot_cohorts:
            print(f"Skipping {cancer_type}: missing hotspot cohort in {args.cancer_organ_map}")
            continue
        output_path = args.output_dir / variant_path.name
        results.append(
            process_cancer_file(
                variant_path,
                driver_path,
                output_path,
                hotspot_cohorts,
                hotspots,
                args.max_driver_genes,
                args.chunk_size,
            )
        )

    for cancer_type, n_drivers, n_variants, n_hotspots, n_observed_genes in results:
        print(
            f"{cancer_type}: driver_genes={n_drivers}, observed_driver_genes={n_observed_genes}, "
            f"variants={n_variants:,}, hotspots={n_hotspots:,}"
        )
    print(f"Wrote {len(results)} cancer variant file(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
