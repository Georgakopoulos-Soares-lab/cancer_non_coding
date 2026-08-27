#!/usr/bin/env python
"""Correlate AlphaGenome RNA-seq disruption measures at patient-gene level."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCORES = (
    REPO_ROOT
    / "data/alphagenome_scores/patient_tissue_tracks_local"
    / "patient_tissue_track_scores.csv"
)
DEFAULT_OUT_DIR = REPO_ROOT / "results/alphagenome_rnaseq_correlation"

GROUP_COLS = ["patient_id", "cancer_type", "gene"]
SCORE_TYPES = ["DIFF_MEAN", "L2_DIFF"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Correlate AlphaGenome DIFF_MEAN and L2_DIFF RNA-seq disruption "
            "scores across matched patient-gene rows."
        )
    )
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--min-n",
        type=int,
        default=3,
        help="Minimum matched patient-gene rows required for a correlation.",
    )
    return parser.parse_args()


def load_rnaseq_scores(path: Path) -> pd.DataFrame:
    columns = [
        "patient_id",
        "cancer_type",
        "gene",
        "gene_strand",
        "output_type",
        "scorer",
        "aggregation_type",
        "score",
        "track_strand",
    ]
    scores = pd.read_csv(path, usecols=columns)
    scores = scores[
        (scores["output_type"] == "RNA_SEQ")
        & (scores["scorer"] == "PATIENT_SINGLE_TRACK_SCORER")
        & scores["aggregation_type"].isin(SCORE_TYPES)
    ].copy()

    strand_match = (
        (scores["track_strand"] == scores["gene_strand"])
        | (scores["track_strand"] == ".")
    )
    scores = scores[strand_match]
    scores["score"] = pd.to_numeric(scores["score"], errors="coerce")
    scores = scores.dropna(subset=["score"])

    patient_gene_scores = (
        scores.groupby([*GROUP_COLS, "aggregation_type"], as_index=False)["score"]
        .mean()
        .pivot(index=GROUP_COLS, columns="aggregation_type", values="score")
        .reset_index()
    )
    patient_gene_scores.columns.name = None
    return patient_gene_scores.dropna(subset=SCORE_TYPES)


def corr_summary(
    frame: pd.DataFrame,
    group_cols: list[str] | None,
    min_n: int,
) -> pd.DataFrame:
    iterator = frame.groupby(group_cols, dropna=False) if group_cols else [((), frame)]
    rows = []
    for key, group in iterator:
        clean = group[SCORE_TYPES].replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean) < min_n:
            continue
        row = {
            "n": len(clean),
            "pearson": clean["DIFF_MEAN"].corr(clean["L2_DIFF"], method="pearson"),
            "spearman": clean["DIFF_MEAN"].corr(clean["L2_DIFF"], method="spearman"),
        }
        if group_cols:
            if not isinstance(key, tuple):
                key = (key,)
            row.update(dict(zip(group_cols, key)))
        rows.append(row)
    return pd.DataFrame(rows)


def save_summary(
    values: pd.DataFrame,
    group_cols: list[str] | None,
    path: Path,
    min_n: int,
) -> pd.DataFrame:
    summary = corr_summary(values, group_cols, min_n)
    if group_cols and not summary.empty:
        summary = summary.sort_values("spearman", ascending=False)
    summary.to_csv(path, sep="\t", index=False)
    return summary


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    values = load_rnaseq_scores(args.scores)
    if values.empty:
        raise ValueError("No matched AlphaGenome RNA-seq disruption scores found.")

    values.to_csv(
        args.out_dir / "patient_gene_rnaseq_disruption_values.tsv",
        sep="\t",
        index=False,
    )
    global_corr = save_summary(
        values,
        None,
        args.out_dir / "global_correlation.tsv",
        args.min_n,
    )
    save_summary(
        values,
        ["patient_id"],
        args.out_dir / "per_patient_correlation.tsv",
        args.min_n,
    )
    save_summary(
        values,
        ["gene"],
        args.out_dir / "per_gene_correlation.tsv",
        args.min_n,
    )
    save_summary(
        values,
        ["cancer_type"],
        args.out_dir / "per_cancer_type_correlation.tsv",
        args.min_n,
    )
    save_summary(
        values,
        ["cancer_type", "gene"],
        args.out_dir / "per_cancer_gene_correlation.tsv",
        args.min_n,
    )

    print(f"Matched patient-gene rows: {len(values):,}")
    if not global_corr.empty:
        row = global_corr.iloc[0]
        print(
            f"Global correlation (n={int(row['n'])}): "
            f"Pearson={row['pearson']:.4f}, Spearman={row['spearman']:.4f}"
        )
    print(f"Wrote results to {args.out_dir}")


if __name__ == "__main__":
    main()
