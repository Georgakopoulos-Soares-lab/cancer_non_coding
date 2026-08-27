#!/usr/bin/env python
"""Map POG570 analysis cohorts to cancer-specific driver gene coordinate files."""

import argparse
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[3]
PATIENT_COHORT_MAPPING = PROJECT_DIR / "metadata" / "pog570_patient_analysis_cohort_mapping.tsv"
DRIVER_GENE_DIR = PROJECT_DIR / "data" / "driver_genes_coords"
COHORT_OUTPUT = PROJECT_DIR / "metadata" / "pog570_cohort_driver_gene_file_mapping.tsv"
PATIENT_OUTPUT = PROJECT_DIR / "metadata" / "pog570_patient_driver_gene_file_mapping.tsv"
COMBINED_OUTPUT_DIR = DRIVER_GENE_DIR / "pog570_cohorts"
EXCLUDED_POG_COHORTS = {"ECR", "SECR", "MISC"}

POG_COHORT_TO_DRIVER_CANCER_TYPES = {
    "BCC": ["SKCM"],
    "CERV": ["CESC"],
    "CNS-PNS": ["GBM", "LGG"],
    "COLO": ["COAD", "READ"],
    "HCC": ["LIHC"],
    "KDNY": ["KICH", "KIRC", "KIRP"],
    "LUNG": ["LUAD", "LUSC"],
    "LYMP": ["DLBC"],
    "PANC": ["PAAD"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient-cohort-mapping", type=Path, default=PATIENT_COHORT_MAPPING)
    parser.add_argument("--driver-gene-dir", type=Path, default=DRIVER_GENE_DIR)
    parser.add_argument("--cohort-output", type=Path, default=COHORT_OUTPUT)
    parser.add_argument("--patient-output", type=Path, default=PATIENT_OUTPUT)
    parser.add_argument("--combined-output-dir", type=Path, default=COMBINED_OUTPUT_DIR)
    parser.add_argument("--write-combined-files", action="store_true")
    return parser.parse_args()


def available_driver_cancer_types(driver_gene_dir: Path) -> set[str]:
    return {
        path.stem
        for path in driver_gene_dir.glob("*.tsv")
        if not path.stem.startswith("Pancancer") and path.stem != "all_driver_genes_hg38"
    }


def driver_cancer_types_for_cohort(cohort: str, available: set[str]) -> list[str]:
    if cohort in POG_COHORT_TO_DRIVER_CANCER_TYPES:
        return POG_COHORT_TO_DRIVER_CANCER_TYPES[cohort]
    if cohort in available:
        return [cohort]
    return []


def write_combined_driver_file(cohort: str, driver_files: list[Path], output_dir: Path) -> Path | None:
    if not driver_files:
        return None

    frames = []
    for path in driver_files:
        df = pd.read_csv(path, sep="\t")
        df.insert(0, "source_driver_cancer_type", path.stem)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    sort_cols = [col for col in ["gene", "chr", "start", "end", "source_driver_cancer_type"] if col in combined.columns]
    combined = combined.drop_duplicates().sort_values(sort_cols)

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{cohort}.tsv"
    combined.to_csv(output, sep="\t", index=False)
    return output


def output_path(path: Path | None) -> str:
    if path is None:
        return ""
    return str(path.resolve().relative_to(PROJECT_DIR))


def main() -> None:
    args = parse_args()
    patients = pd.read_csv(args.patient_cohort_mapping, sep="\t", dtype=str)
    patients = patients[~patients["analysis_cohort"].isin(EXCLUDED_POG_COHORTS)].copy()
    available = available_driver_cancer_types(args.driver_gene_dir)

    rows = []
    for cohort, group in patients.groupby("analysis_cohort", dropna=False):
        driver_cancer_types = driver_cancer_types_for_cohort(cohort, available)
        missing = [name for name in driver_cancer_types if name not in available]
        driver_cancer_types = [name for name in driver_cancer_types if name in available]
        driver_files = [args.driver_gene_dir / f"{name}.tsv" for name in driver_cancer_types]
        combined_file = None
        if args.write_combined_files:
            combined_file = write_combined_driver_file(cohort, driver_files, args.combined_output_dir)

        rows.append({
            "analysis_cohort": cohort,
            "n_patients": group["patient_id"].nunique(),
            "driver_cancer_types": ";".join(driver_cancer_types),
            "driver_gene_files": ";".join(output_path(path) for path in driver_files),
            "combined_driver_gene_file": output_path(combined_file),
            "n_driver_files": len(driver_files),
            "missing_driver_cancer_types": ";".join(missing),
            "has_driver_gene_file": bool(driver_files),
        })

    cohort_mapping = pd.DataFrame(rows).sort_values("analysis_cohort")
    patient_mapping = patients.merge(cohort_mapping, on="analysis_cohort", how="left")

    args.cohort_output.parent.mkdir(parents=True, exist_ok=True)
    args.patient_output.parent.mkdir(parents=True, exist_ok=True)
    cohort_mapping.to_csv(args.cohort_output, sep="\t", index=False)
    patient_mapping.to_csv(args.patient_output, sep="\t", index=False)

    print(f"Wrote {len(cohort_mapping)} cohort mappings to {args.cohort_output}")
    print(f"Wrote {len(patient_mapping)} patient mappings to {args.patient_output}")
    unresolved = cohort_mapping.loc[~cohort_mapping["has_driver_gene_file"], "analysis_cohort"].tolist()
    if unresolved:
        print("Unresolved cohorts: " + ", ".join(unresolved))


if __name__ == "__main__":
    main()
