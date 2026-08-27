#!/usr/bin/env python
"""List TCGA cancer types by number of matched AlphaGenome output modalities.

Cancer types are matched to AlphaGenome biosamples using keyword values from
metadata/tcga_cancer_tissue_cell_lines.tsv.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_METADATA_PATH = PROJECT_DIR / "metadata" / "alphagenome_metadata.csv"
DEFAULT_TISSUE_METADATA_PATH = PROJECT_DIR / "metadata" / "tcga_cancer_tissue_cell_lines.tsv"
DEFAULT_MIN_MODALITIES = 1
DEFAULT_TOP = None

OUTPUT_COLUMNS = [
    "cancer_type",
    "n_modalities",
    "n_tracks",
]


def normalize_output_type(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    return text.rsplit(".", 1)[-1].upper()


def summarize_cancer_type_modalities(
    metadata_path: Path,
    tissue_metadata_path: Path,
    min_modalities: int,
) -> tuple[pd.DataFrame, int]:
    df = pd.read_csv(metadata_path, dtype=str)
    required = {"output_type", "biosample_name"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required metadata columns: {missing}")

    tissue_df = pd.read_csv(tissue_metadata_path, sep="\t", dtype=str)
    tissue_required = {"cancer_type", "keyword"}
    tissue_missing = sorted(tissue_required - set(tissue_df.columns))
    if tissue_missing:
        raise ValueError(f"Missing required tissue metadata columns: {tissue_missing}")

    df = df.copy()
    df["modality"] = df["output_type"].map(normalize_output_type)
    df = df[df["modality"] != ""]

    all_modalities = sorted(df["modality"].dropna().unique())
    total_modalities = len(all_modalities)

    df = df[df["biosample_name"].notna() & (df["biosample_name"].str.strip() != "")]
    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS), total_modalities
    df["biosample_name_lower"] = df["biosample_name"].str.lower()

    rows = []
    for _, tissue_row in tissue_df.iterrows():
        cancer_type = str(tissue_row["cancer_type"]).strip()
        keywords = [
            keyword.strip().lower()
            for keyword in str(tissue_row["keyword"]).split(",")
            if keyword.strip()
        ]
        if not cancer_type or not keywords:
            continue

        mask = df["biosample_name_lower"].map(
            lambda biosample_name: any(keyword in biosample_name for keyword in keywords)
        )
        matched = df[mask]
        rows.append(
            {
                "cancer_type": cancer_type,
                "n_modalities": matched["modality"].nunique(),
                "n_tracks": len(matched),
            }
        )

    grouped = pd.DataFrame(rows)
    if grouped.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS), total_modalities

    all_modalities_set = set(all_modalities)
    grouped = grouped[grouped["n_modalities"] >= min_modalities]
    grouped = grouped.sort_values(
        ["n_modalities", "n_tracks", "cancer_type"],
        ascending=[False, False, True],
    )
    return grouped[OUTPUT_COLUMNS].reset_index(drop=True), total_modalities


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Print TCGA cancer types sorted by how many AlphaGenome output "
            "modalities have at least one keyword-matched track."
        )
    )
    parser.add_argument(
        "--metadata",
        default=str(DEFAULT_METADATA_PATH),
        help=f"AlphaGenome metadata CSV (default: {DEFAULT_METADATA_PATH}).",
    )
    parser.add_argument(
        "--tissue-metadata",
        default=str(DEFAULT_TISSUE_METADATA_PATH),
        help=f"TCGA cancer tissue keyword TSV (default: {DEFAULT_TISSUE_METADATA_PATH}).",
    )
    parser.add_argument(
        "--min-modalities",
        type=int,
        default=DEFAULT_MIN_MODALITIES,
        help="Minimum number of modalities to print (default: %(default)s).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=DEFAULT_TOP,
        help="Print only the top N rows.",
    )
    args = parser.parse_args()

    if args.min_modalities < 1:
        parser.error("--min-modalities must be a positive integer.")
    if args.top is not None and args.top < 1:
        parser.error("--top must be a positive integer.")

    summary, total_modalities = summarize_cancer_type_modalities(
        metadata_path=Path(args.metadata),
        tissue_metadata_path=Path(args.tissue_metadata),
        min_modalities=args.min_modalities,
    )
    if args.top is not None:
        summary = summary.head(args.top)

    print(f"total_modalities\t{total_modalities}")
    summary.to_csv(sys.stdout, sep="\t", index=False)


if __name__ == "__main__":
    main()
