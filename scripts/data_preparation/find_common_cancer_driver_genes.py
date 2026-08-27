#!/usr/bin/env python3
"""Find genes appearing in the top N drivers of the most cancer types.

Example:
    python scripts/data_preparation/find_common_cancer_driver_genes.py --top-n 10
"""

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]


def cancer_driver_files(directory):
    return sorted(
        path
        for path in directory.glob("*.tsv")
        if not path.stem.startswith("Pancancer") and path.stem != "all_driver_genes_hg38"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument(
        "--driver-genes-dir",
        type=Path,
        default=REPO_ROOT / "data/driver_genes_coords",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.top_n < 1:
        raise ValueError("--top-n must be at least 1")

    frames = []
    for path in cancer_driver_files(args.driver_genes_dir):
        genes = pd.read_csv(path, sep="\t", usecols=["gene"])["gene"].dropna().head(args.top_n)
        frames.append(pd.DataFrame({"gene": genes, "cancer_type": path.stem}))

    common_genes = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates()
        .groupby("gene", as_index=False)
        .agg(
            n_cancer_types=("cancer_type", "nunique"),
            cancer_types=("cancer_type", lambda values: ",".join(sorted(values))),
        )
        .sort_values(["n_cancer_types", "gene"], ascending=[False, True])
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        common_genes.to_csv(args.output, sep="\t", index=False)
        print(f"Saved {len(common_genes):,} genes to {args.output}")
    else:
        print(common_genes.to_csv(sep="\t", index=False), end="")


if __name__ == "__main__":
    main()
