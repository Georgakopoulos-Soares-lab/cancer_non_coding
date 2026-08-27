#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Find driver genes shared across all or most cancer-type files in "
            "data/driver_genes_coords."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=root / "data" / "driver_genes_coords",
        help="Directory containing per-cancer driver gene TSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "results" / "driver_gene_overlap",
        help="Directory for output files.",
    )
    parser.add_argument(
        "--min-fraction",
        type=float,
        default=0.8,
        help=(
            "Minimum fraction of cancer types for the 'most cancer types' list. "
            "Example: 0.8 means at least 80 percent of cancer types."
        ),
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=None,
        help=(
            "Optional absolute minimum number of cancer types for the 'most' list. "
            "If omitted, it is derived from --min-fraction."
        ),
    )
    parser.add_argument(
        "--include-pancancer",
        action="store_true",
        help="Include Pancancer.tsv as an additional cancer type.",
    )
    parser.add_argument(
        "--include-all-driver-file",
        action="store_true",
        help="Include all_driver_genes_hg38.tsv as an additional set.",
    )
    return parser.parse_args()


def _read_gene_set(tsv_path: Path) -> Set[str]:
    genes: Set[str] = set()
    with tsv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {tsv_path}")
        if "gene" not in reader.fieldnames:
            raise ValueError(f"Expected a 'gene' column in {tsv_path}")

        for row in reader:
            gene = (row.get("gene") or "").strip()
            if gene:
                genes.add(gene)
    return genes


def _discover_files(
    input_dir: Path,
    include_pancancer: bool,
    include_all_driver_file: bool,
) -> List[Tuple[str, Path]]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    out: List[Tuple[str, Path]] = []
    for path in sorted(input_dir.glob("*.tsv")):
        stem = path.stem

        if stem == "all_driver_genes_hg38" and not include_all_driver_file:
            continue
        if stem == "Pancancer" and not include_pancancer:
            continue

        out.append((stem, path))

    if not out:
        raise ValueError("No TSV files selected. Check input directory and include flags.")

    return out


def _compute_recurrence(gene_sets: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    recurrence: Dict[str, Set[str]] = {}
    for cancer_type, genes in gene_sets.items():
        for gene in genes:
            recurrence.setdefault(gene, set()).add(cancer_type)
    return recurrence


def _write_gene_list(path: Path, genes: List[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("gene\n")
        for gene in genes:
            f.write(f"{gene}\n")


def _write_recurrence_table(path: Path, recurrence_rows: List[Tuple[str, int, float, str]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("gene\tcancer_type_count\tcancer_type_fraction\tcancer_types\n")
        for gene, count, frac, cancer_types in recurrence_rows:
            f.write(f"{gene}\t{count}\t{frac:.6f}\t{cancer_types}\n")


def main() -> None:
    args = parse_args()

    if args.min_fraction <= 0.0 or args.min_fraction > 1.0:
        raise ValueError("--min-fraction must be in the interval (0, 1].")

    files = _discover_files(
        input_dir=args.input_dir.resolve(),
        include_pancancer=args.include_pancancer,
        include_all_driver_file=args.include_all_driver_file,
    )

    gene_sets: Dict[str, Set[str]] = {}
    for cancer_type, file_path in files:
        gene_sets[cancer_type] = _read_gene_set(file_path)

    n_types = len(gene_sets)
    if n_types == 0:
        raise ValueError("No cancer types available after filtering.")

    recurrence = _compute_recurrence(gene_sets)

    all_common = sorted([g for g, cancer_types in recurrence.items() if len(cancer_types) == n_types])

    min_count = args.min_count
    if min_count is None:
        min_count = max(1, int(round(args.min_fraction * n_types)))
    min_count = max(1, min(min_count, n_types))

    most_common = sorted([g for g, cancer_types in recurrence.items() if len(cancer_types) >= min_count])

    recurrence_rows = sorted(
        (
            (
                g,
                len(cancer_types),
                len(cancer_types) / n_types,
                ",".join(sorted(cancer_types)),
            )
            for g, cancer_types in recurrence.items()
        ),
        key=lambda x: (-x[1], x[0]),
    )

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cancer_types_path = out_dir / "cancer_types_used.txt"
    with cancer_types_path.open("w", encoding="utf-8") as f:
        for cancer_type in sorted(gene_sets):
            f.write(f"{cancer_type}\n")

    all_common_path = out_dir / "genes_common_all_cancer_types.tsv"
    most_common_path = out_dir / "genes_common_most_cancer_types.tsv"
    recurrence_path = out_dir / "gene_recurrence_by_cancer_type.tsv"
    summary_path = out_dir / "summary.json"

    _write_gene_list(all_common_path, all_common)
    _write_gene_list(most_common_path, most_common)
    _write_recurrence_table(recurrence_path, recurrence_rows)

    summary = {
        "input_dir": str(args.input_dir.resolve()),
        "n_cancer_types": n_types,
        "min_fraction": args.min_fraction,
        "min_count": min_count,
        "n_genes_common_all": len(all_common),
        "n_genes_common_most": len(most_common),
        "output_files": {
            "cancer_types": str(cancer_types_path),
            "all_common": str(all_common_path),
            "most_common": str(most_common_path),
            "recurrence_table": str(recurrence_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("Driver gene overlap analysis complete")
    print(f"Cancer types used: {n_types}")
    print(f"Common to all cancer types: {len(all_common)} genes")
    print(
        "Common to most cancer types: "
        f"{len(most_common)} genes (threshold: >= {min_count}/{n_types})"
    )
    print(f"Saved outputs under: {out_dir}")


if __name__ == "__main__":
    main()
