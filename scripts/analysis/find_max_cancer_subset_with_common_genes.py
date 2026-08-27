#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Set, Tuple


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Find the largest subset of cancer types whose shared driver-gene "
            "intersection has at least N genes."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=root / "data" / "driver_genes_coords",
        help="Directory with per-cancer TSV files containing a gene column.",
    )
    parser.add_argument(
        "--min-common-genes",
        type=int,
        default=5,
        help="Minimum number of genes required in the cancer-set intersection.",
    )
    parser.add_argument(
        "--include-pancancer",
        action="store_true",
        help="Include Pancancer.tsv as a cancer type.",
    )
    parser.add_argument(
        "--include-all-driver-file",
        action="store_true",
        help="Include all_driver_genes_hg38.tsv as a cancer type.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "results" / "driver_gene_overlap",
        help="Directory to save the computed subset/common-gene outputs.",
    )
    return parser.parse_args()


def load_sets(input_dir: Path, include_pancancer: bool, include_all_driver_file: bool) -> Tuple[List[str], List[Set[str]]]:
    files = []
    for p in sorted(input_dir.glob("*.tsv")):
        if p.stem == "Pancancer" and not include_pancancer:
            continue
        if p.stem == "all_driver_genes_hg38" and not include_all_driver_file:
            continue
        files.append(p)

    cancers: List[str] = []
    sets: List[Set[str]] = []
    for p in files:
        with p.open("r", encoding="utf-8") as f:
            r = csv.DictReader(f, delimiter="\t")
            if r.fieldnames is None or "gene" not in r.fieldnames:
                raise ValueError(f"Missing 'gene' column in {p}")
            genes = {(row.get("gene") or "").strip() for row in r}
            genes.discard("")
        cancers.append(p.stem)
        sets.append(genes)

    if not cancers:
        raise ValueError("No input cancer files found.")
    return cancers, sets


def main() -> None:
    args = parse_args()
    if args.min_common_genes <= 0:
        raise ValueError("--min-common-genes must be > 0")

    cancers, gene_sets = load_sets(
        args.input_dir.resolve(),
        include_pancancer=args.include_pancancer,
        include_all_driver_file=args.include_all_driver_file,
    )

    n = len(cancers)
    threshold = args.min_common_genes

    # Reorder to improve pruning (start with smaller sets).
    order = sorted(range(n), key=lambda i: len(gene_sets[i]))
    cancers_ord = [cancers[i] for i in order]
    sets_ord = [gene_sets[i] for i in order]

    best_size = 0
    best_idx: List[int] = []
    best_common: Set[str] = set()

    def dfs(i: int, chosen: List[int], common: Set[str]) -> None:
        nonlocal best_size, best_idx, best_common

        remaining = n - i
        if len(chosen) + remaining <= best_size:
            return

        if len(chosen) > best_size and len(common) >= threshold:
            best_size = len(chosen)
            best_idx = chosen.copy()
            best_common = common.copy()

        if i >= n:
            return

        # Include branch.
        new_common = common & sets_ord[i]
        if len(new_common) >= threshold:
            chosen.append(i)
            dfs(i + 1, chosen, new_common)
            chosen.pop()

        # Exclude branch.
        dfs(i + 1, chosen, common)

    all_genes = set.union(*sets_ord)
    dfs(0, [], all_genes)

    best_cancers = sorted(cancers_ord[i] for i in best_idx)
    best_genes = sorted(best_common)

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    output_stem = f"max_subset_min_common_{threshold}"
    summary_path = out_dir / f"{output_stem}.json"
    cancers_path = out_dir / f"{output_stem}_cancer_types.txt"
    genes_path = out_dir / f"{output_stem}_common_genes.txt"

    summary = {
        "input_dir": str(args.input_dir.resolve()),
        "min_common_genes": threshold,
        "n_cancer_types_total": n,
        "largest_subset_size": best_size,
        "largest_subset_cancer_types": best_cancers,
        "n_common_genes": len(best_genes),
        "common_genes": best_genes,
        "include_pancancer": args.include_pancancer,
        "include_all_driver_file": args.include_all_driver_file,
        "output_files": {
            "summary_json": str(summary_path),
            "cancer_types_txt": str(cancers_path),
            "common_genes_txt": str(genes_path),
        },
    }

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    cancers_path.write_text("\n".join(best_cancers) + ("\n" if best_cancers else ""), encoding="utf-8")
    genes_path.write_text("\n".join(best_genes) + ("\n" if best_genes else ""), encoding="utf-8")

    print(f"Total cancer types considered: {n}")
    print(f"Threshold (min common genes): {threshold}")
    print(f"Largest cancer-type subset size: {best_size}")
    print("Cancer types in largest subset:")
    print(",".join(best_cancers))
    print(f"Number of common genes in this subset: {len(best_genes)}")
    print("Common genes:")
    print(",".join(best_genes))
    print(f"Saved outputs under: {out_dir}")


if __name__ == "__main__":
    main()
