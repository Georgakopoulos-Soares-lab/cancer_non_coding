#!/usr/bin/env python
"""
Count saved hg38 hotspot mutations in each patient-level TCGA variant file.

The matcher uses:
  - patient variants from data/TCGA/tcga_patient_variants/*.tsv
  - saved hg38 hotspots from data/TCGA/pancancer_driver_hotspots_hg38.tsv
  - target driver genes from data/driver_genes_coords/Pancancer_1pc.tsv

Output is a patient-level CSV with counts of matching hotspot mutation rows,
unique mutation-gene pairs, unique mutation IDs, and unique hotspot genes.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


DEFAULT_INPUT_DIR = Path("data/TCGA/tcga_patient_variants")
DEFAULT_HOTSPOT_FILE = Path("data/TCGA/pancancer_driver_hotspots_hg38.tsv")
DEFAULT_DRIVER_GENE_FILE = Path("data/driver_genes_coords/Pancancer_1pc.tsv")
DEFAULT_OUTPUT_CSV = Path("results/patient_hotspot_mutation_counts.csv")


def normalize_chromosome(chrom: object) -> str | None:
    if chrom is None:
        return None
    chrom_s = str(chrom).strip()
    if not chrom_s or chrom_s.lower() == "nan":
        return None
    if chrom_s.lower().startswith("chr"):
        chrom_s = chrom_s[3:]
    return chrom_s


def extract_gene_symbols(gene_value: object) -> set[str]:
    if not isinstance(gene_value, str):
        return set()
    symbols: set[str] = set()
    for token in gene_value.replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        token = token.split("(", 1)[0].strip()
        if token and token.upper() not in {"NONE", "NA", "NAN", "."}:
            symbols.add(token)
    return symbols


def normalise_mutation(mutation: str, indexing: str = "1-based") -> str:
    parts = str(mutation).split(":")
    if len(parts) != 4:
        return str(mutation)
    chrom, coord, ref, alt = parts
    if "-" not in coord:
        pos = int(coord)
        coord = f"{pos}-{pos + 1}" if indexing == "0-based" else f"{pos}-{pos}"
    return f"{chrom}:{coord}:{ref}:{alt}"


def parse_mutation_interval(mutation: str, indexing: str) -> tuple[str, int, int]:
    chrom, coord, _ref, _alt = mutation.split(":", 3)
    start_raw, end_raw = map(int, coord.split("-"))
    low = min(start_raw, end_raw)
    high = max(start_raw, end_raw)
    if indexing == "1-based":
        start1 = low
        end1 = high
    else:
        start1 = low + 1
        end1 = max(start1, high)
    chrom = normalize_chromosome(chrom)
    if chrom is None:
        raise ValueError(f"missing chromosome in mutation: {mutation}")
    return chrom, start1, end1


def find_column(fieldnames: list[str], candidates: tuple[str, ...]) -> str:
    by_lower = {field.lower(): field for field in fieldnames}
    for candidate in candidates:
        if candidate in by_lower:
            return by_lower[candidate]
    raise ValueError(f"Missing required column. Expected one of: {', '.join(candidates)}")


def load_driver_genes(path: Path) -> set[str]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if "gene" not in (reader.fieldnames or []):
            raise ValueError(f"{path} missing required column: gene")
        return {row["gene"].strip() for row in reader if row.get("gene", "").strip()}


def load_hotspots(path: Path, allowed_genes: set[str]) -> dict[str, dict[str, set[int]]]:
    hotspots: dict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"chromosome", "start", "gene"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing required column(s): {sorted(missing)}")
        for row in reader:
            gene = row["gene"].strip()
            if gene not in allowed_genes:
                continue
            chrom = normalize_chromosome(row["chromosome"])
            if chrom is None:
                continue
            try:
                start = int(row["start"])
            except ValueError:
                continue
            hotspots[gene][chrom].add(start)
    return {gene: dict(by_chrom) for gene, by_chrom in hotspots.items()}


def count_patient_file(
    patient_file: Path,
    hotspots: dict[str, dict[str, set[int]]],
    allowed_genes: set[str],
    indexing: str,
) -> dict[str, object]:
    with patient_file.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = reader.fieldnames or []
        mutation_col = find_column(fieldnames, ("mutation", "id", "variant"))
        gene_col = find_column(fieldnames, ("gene", "gene_name"))

        total_rows = 0
        parseable_rows = 0
        rows_with_driver_gene = 0
        rows_with_hotspot_gene = 0
        hotspot_rows = 0
        hotspot_mutation_gene_pairs: set[tuple[str, str]] = set()
        hotspot_mutations: set[str] = set()
        hotspot_genes: set[str] = set()

        for row in reader:
            total_rows += 1
            mutation_raw = row.get(mutation_col, "")
            try:
                mutation = normalise_mutation(mutation_raw, indexing=indexing)
                chrom, start1, end1 = parse_mutation_interval(mutation, indexing=indexing)
            except Exception:
                continue
            parseable_rows += 1

            driver_genes = extract_gene_symbols(row.get(gene_col, "")).intersection(allowed_genes)
            if not driver_genes:
                continue
            rows_with_driver_gene += 1

            candidate_genes = driver_genes.intersection(hotspots.keys())
            if not candidate_genes:
                continue
            rows_with_hotspot_gene += 1

            matched_genes = []
            for gene in candidate_genes:
                starts = hotspots.get(gene, {}).get(chrom, set())
                if any(start1 <= pos <= end1 for pos in starts):
                    matched_genes.append(gene)

            if not matched_genes:
                continue

            hotspot_rows += 1
            hotspot_mutations.add(mutation)
            for gene in matched_genes:
                hotspot_genes.add(gene)
                hotspot_mutation_gene_pairs.add((mutation, gene))

    return {
        "patient_id": patient_file.stem,
        "total_variant_rows": total_rows,
        "parseable_variant_rows": parseable_rows,
        "rows_with_driver_gene": rows_with_driver_gene,
        "rows_with_hotspot_gene": rows_with_hotspot_gene,
        "hotspot_variant_rows": hotspot_rows,
        "unique_hotspot_mutations": len(hotspot_mutations),
        "unique_hotspot_mutation_gene_pairs": len(hotspot_mutation_gene_pairs),
        "unique_hotspot_genes": len(hotspot_genes),
        "hotspot_genes": ";".join(sorted(hotspot_genes)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count saved hg38 hotspot mutations in each patient-level variant TSV."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--hotspot-hg38-file", type=Path, default=DEFAULT_HOTSPOT_FILE)
    parser.add_argument("--driver-gene-file", type=Path, default=DEFAULT_DRIVER_GENE_FILE)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--patient-list", type=Path, default=None, help="Optional file with one patient ID per line.")
    parser.add_argument("--indexing", choices=["0-based", "1-based"], default="1-based")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    allowed_genes = load_driver_genes(args.driver_gene_file)
    hotspots = load_hotspots(args.hotspot_hg38_file, allowed_genes)
    hotspot_genes = set(hotspots)

    if args.patient_list:
        patient_ids = [
            line.strip()
            for line in args.patient_list.read_text().splitlines()
            if line.strip()
        ]
        patient_files = [args.input_dir / f"{patient_id}.tsv" for patient_id in patient_ids]
    else:
        patient_files = sorted(args.input_dir.glob("*.tsv"))

    missing = [path for path in patient_files if not path.exists()]
    if missing:
        preview = ", ".join(str(path) for path in missing[:5])
        raise FileNotFoundError(f"{len(missing)} patient TSV(s) missing. First missing: {preview}")

    print(
        f"Counting hotspots for {len(patient_files)} patient(s); "
        f"{len(allowed_genes)} target gene(s), {len(hotspot_genes)} with saved hotspots."
    )

    rows = [
        count_patient_file(path, hotspots=hotspots, allowed_genes=allowed_genes, indexing=args.indexing)
        for path in patient_files
    ]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "patient_id",
        "total_variant_rows",
        "parseable_variant_rows",
        "rows_with_driver_gene",
        "rows_with_hotspot_gene",
        "hotspot_variant_rows",
        "unique_hotspot_mutations",
        "unique_hotspot_mutation_gene_pairs",
        "unique_hotspot_genes",
        "hotspot_genes",
    ]
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_with_hotspots = sum(int(row["unique_hotspot_mutation_gene_pairs"]) > 0 for row in rows)
    total_pairs = sum(int(row["unique_hotspot_mutation_gene_pairs"]) for row in rows)
    print(f"Saved: {args.output_csv}")
    print(f"Patients with >=1 hotspot mutation-gene pair: {n_with_hotspots} / {len(rows)}")
    print(f"Total unique hotspot mutation-gene pairs: {total_pairs}")


if __name__ == "__main__":
    main()
