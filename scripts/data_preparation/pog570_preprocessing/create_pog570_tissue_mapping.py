#!/usr/bin/env python
"""Create POG570 analysis-cohort and patient-cohort mapping files."""

import argparse
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[3]
POG570_MUTATIONS = PROJECT_DIR / "data" / "POG570" / "POG570_small_mutations.txt"
ALPHAGENOME_TRACKS = PROJECT_DIR / "metadata" / "tissue_alphagenome_tracks.tsv"
OUTPUT = PROJECT_DIR / "metadata" / "pog570_tissue_mapping.tsv"
PATIENT_OUTPUT = PROJECT_DIR / "metadata" / "pog570_patient_analysis_cohort_mapping.tsv"

COHORT_TO_TISSUE = {
    "ACC": "Adrenal Gland",
    "BCC": "Skin",
    "BLCA": "Bladder",
    "BRCA": "Breast",
    "CERV": "Cervix",
    "CHOL": "Biliary",
    "CNS-PNS": "CNS",
    "COLO": "Colon",
    "ESCA": "Esophagus",
    "HCC": "Liver",
    "HNSC": "Head",
    "KDNY": "Kidney",
    "LUNG": "Lung",
    "LYMP": "Lymphoid",
    "OV": "Ovary",
    "PANC": "Pancreas",
    "PRAD": "Prostate",
    "SARC": "Sarcoma",
    "SKCM": "Skin",
    "STAD": "Stomach",
    "THCA": "Thyroid",
    "THYM": "Thymus",
    "UCEC": "Uterus",
    "UVM": "Eye",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create POG570 AlphaGenome tissue mapping TSV.")
    parser.add_argument("--pog570-mutations", type=Path, default=POG570_MUTATIONS)
    parser.add_argument("--alphagenome-tracks", type=Path, default=ALPHAGENOME_TRACKS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--patient-output", type=Path, default=PATIENT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    patients = pd.read_csv(args.pog570_mutations, sep="\t", dtype=str)[
        ["patient_id", "analysis_cohort"]
    ].drop_duplicates("patient_id")
    patients["alphagenome_tissue"] = patients["analysis_cohort"].map(COHORT_TO_TISSUE)

    tracks = pd.read_csv(args.alphagenome_tracks, sep="\t", usecols=["tissue"])
    track_counts = tracks["tissue"].value_counts().rename_axis("alphagenome_tissue").reset_index(name="n_tracks")

    mapping = (
        patients.groupby(["analysis_cohort", "alphagenome_tissue"], dropna=False, as_index=False)
        .agg(n_patients=("patient_id", "nunique"))
        .merge(track_counts, on="alphagenome_tissue", how="left")
        .sort_values("analysis_cohort")
    )
    mapping["n_tracks"] = mapping["n_tracks"].fillna(0).astype(int)
    mapping["has_tracks"] = mapping["n_tracks"] > 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(args.output, sep="\t", index=False)
    patients[["patient_id", "analysis_cohort"]].sort_values("patient_id").to_csv(
        args.patient_output,
        sep="\t",
        index=False,
    )
    print(f"Wrote {len(mapping)} analysis-cohort mappings to {args.output}")
    print(f"Wrote {len(patients)} patient-cohort mappings to {args.patient_output}")


if __name__ == "__main__":
    main()
