"""
Count the number of TCGA patients that have at least one mutation in each
driver gene listed in data/driver_genes_intogen_curated/Pancancer.tsv.

Patient mutation data: data/TCGA/tcga_patient_variants/<patient_id>.tsv
Each patient file has a 'gene' column whose entries may look like:
  - TP53
  - TNFRSF14(NM_003820:c.*473C>T,NM_001297605:c.*627C>T)
  - LINC01777(dist=51714),LINC01646(dist=95089)

Gene names are extracted as the identifier before the first '(' in each
comma-separated top-level token.
"""

import os
import tqdm
import csv
import pandas as pd
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


DRIVER_GENES_FILE = Path("data/driver_genes_intogen_curated/Pancancer.tsv")
VARIANTS_DIR = Path("data/TCGA/tcga_patient_variants")
OUTPUT_FILE = Path("results/driver_gene_patient_counts.tsv")
PATIENT_OUTPUT_FILE = Path("results/patient_driver_gene_counts.tsv")


def extract_genes(gene_field: str) -> set[str]:
    """
    Parse the 'gene' field from a variant row and return all gene names.

    Handles formats like:
      PRDM16
      TNFRSF14(NM_003820:c.*473C>T,NM_001297605:c.*627C>T)
      LINC01777(dist=51714),LINC01646(dist=95089)
    """
    genes: set[str] = set()
    # Split by commas that are NOT inside parentheses
    depth = 0
    current: list[str] = []
    tokens: list[str] = []
    for ch in gene_field:
        if ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            tokens.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        tokens.append("".join(current).strip())

    for token in tokens:
        gene_name = token.split("(")[0].strip()
        if gene_name:
            genes.add(gene_name)
    return genes


def load_driver_genes(path: Path) -> list[str]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [row["Symbol"] for row in reader if row.get("Symbol")]


def process_patient(patient_file: Path, driver_gene_set: set[str]) -> tuple[str, set[str]]:
    """Return (patient_id, set of driver genes mutated in this patient)."""
    patient_id = patient_file.stem
    mutated_genes: set[str] = set()
    with open(patient_file, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            gene_field = row.get("gene", "").strip()
            if gene_field:
                mutated_genes.update(extract_genes(gene_field))
    return patient_id, mutated_genes & driver_gene_set


def main():
    # Load driver genes
    driver_genes = load_driver_genes(DRIVER_GENES_FILE)
    driver_gene_set = set(driver_genes)
    print(f"Loaded {len(driver_genes)} driver genes from {DRIVER_GENES_FILE}")

    # Count patients per driver gene
    patient_counts: dict[str, set[str]] = defaultdict(set)
    patient_driver_counts: dict[str, int] = {}

    patient_files = sorted(VARIANTS_DIR.glob("*.tsv"))
    print(f"Processing {len(patient_files)} patient files...")

    n_workers = min(os.cpu_count() or 1, len(patient_files))
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_patient, pf, driver_gene_set): pf for pf in patient_files}
        for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Processing patients"):
            patient_id, driver_hits = future.result()
            for gene in driver_hits:
                patient_counts[gene].add(patient_id)
            patient_driver_counts[patient_id] = len(driver_hits)

    # --- Per-gene output ---
    patient_counts_df = pd.DataFrame({
        "gene": list(patient_counts.keys()),
        "n_patients": [len(patients) for patients in patient_counts.values()]
    }).sort_values("n_patients", ascending=False)
    patient_counts_df.to_csv(OUTPUT_FILE, sep="\t", index=False)
    print(f"Results written to {OUTPUT_FILE}")

    # --- Per-patient output ---
    patient_driver_counts_df = pd.DataFrame({
        "patient_id": list(patient_driver_counts.keys()),
        "n_driver_genes_mutated": list(patient_driver_counts.values())
    }).sort_values("n_driver_genes_mutated", ascending=False)
    patient_driver_counts_df.to_csv(PATIENT_OUTPUT_FILE, sep="\t", index=False)
    print(f"Per-patient results written to {PATIENT_OUTPUT_FILE}")

    # Print a quick summary
    print("\nTop 20 driver genes by patient count:")
    print(f"{'Gene':<20} {'Patients':>10}")
    print("-" * 32)
    sorted_genes = sorted(driver_genes, key=lambda g: len(patient_counts.get(g, set())), reverse=True)
    for gene in sorted_genes[:20]:
        print(f"{gene:<20} {len(patient_counts.get(gene, set())):>10}")


if __name__ == "__main__":
    main()
