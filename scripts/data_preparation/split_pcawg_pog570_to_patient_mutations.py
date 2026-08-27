"""
Split PCAWG and POG570 mutation data into per-sample TSV files with
"mutation" and "gene" columns only.

PCAWG input : data/PCAWG/annotated_snv_mv_indels_by_cancer_subtype/*.tsv
              Columns used: Tumor_Sample_Barcode, mutation, gene
              mutation format is already <chr>:<pos>:<ref>:<alt>

POG570 input: data/POG570/POG570_small_mutations.txt
              Columns used: patient_id, chrom, pos, ref, alt, gene_id
              mutation is built as <chrom>:<pos>:<ref>:<alt>

Output:
    data/PCAWG_POG570_mutations/<Tumor_Sample_Barcode>.tsv
    data/PCAWG_POG570_mutations/<patient_id>.tsv

Usage:
    python scripts/1_split_pcawg_pog570_to_patient_mutations.py
    python scripts/1_split_pcawg_pog570_to_patient_mutations.py \
        --pcawg_dir  data/PCAWG/annotated_snv_mv_indels_by_cancer_subtype \
        --pcawg_out  data/PCAWG_POG570_mutations \
        --pog570     data/POG570/POG570_small_mutations.txt \
        --pog570_out data/PCAWG_POG570_mutations
"""

import argparse
import os
from glob import glob

import pandas as pd
from tqdm import tqdm


PCAWG_DIR  = "data/PCAWG/annotated_snv_mv_indels_by_cancer_subtype"
PCAWG_OUT  = "data/PCAWG_POG570_mutations"
POG570_IN  = "data/POG570/POG570_small_mutations.txt"
POG570_OUT = "data/PCAWG_POG570_mutations"
CHUNK_SIZE = 200_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split PCAWG and POG570 data into per-sample mutation TSVs"
    )
    parser.add_argument("--pcawg_dir",  default=PCAWG_DIR)
    parser.add_argument("--pcawg_out",  default=PCAWG_OUT)
    parser.add_argument("--pog570",     default=POG570_IN)
    parser.add_argument("--pog570_out", default=POG570_OUT)
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_sample_files(sample_rows: dict[str, list[dict]], output_dir: str) -> None:
    """Write accumulated per-sample rows as TSV files."""
    for sample_id, rows in tqdm(sample_rows.items(), desc="Writing", leave=False):
        df = pd.DataFrame(rows, columns=["mutation", "gene"])
        out_path = os.path.join(output_dir, f"{sample_id}.tsv")
        df.to_csv(out_path, sep="\t", index=False)


# ---------------------------------------------------------------------------
# PCAWG
# ---------------------------------------------------------------------------

def process_pcawg(pcawg_dir: str, output_dir: str, chunk_size: int) -> None:
    tsv_files = sorted(glob(os.path.join(pcawg_dir, "*.tsv")))
    if not tsv_files:
        raise FileNotFoundError(f"No TSV files found in {pcawg_dir}")

    os.makedirs(output_dir, exist_ok=True)
    sample_rows: dict[str, list[dict]] = {}

    print(f"\n[PCAWG] Processing {len(tsv_files)} cancer-subtype files...")
    for tsv_path in tqdm(tsv_files, desc="PCAWG files"):
        reader = pd.read_csv(
            tsv_path,
            sep="\t",
            usecols=["Tumor_Sample_Barcode", "mutation", "gene"],
            dtype=str,
            chunksize=chunk_size,
        )
        for chunk in reader:
            chunk = chunk.dropna(subset=["Tumor_Sample_Barcode", "mutation"])
            for _, row in chunk.iterrows():
                sample_id = str(row["Tumor_Sample_Barcode"]).strip()
                if sample_id not in sample_rows:
                    sample_rows[sample_id] = []
                sample_rows[sample_id].append({
                    "mutation": str(row["mutation"]).strip(),
                    "gene":     str(row["gene"]).strip(),
                })

    print(f"[PCAWG] Writing {len(sample_rows):,} sample files to {output_dir}/...")
    write_sample_files(sample_rows, output_dir)
    print(f"[PCAWG] Done.")


# ---------------------------------------------------------------------------
# POG570
# ---------------------------------------------------------------------------

def process_pog570(input_path: str, output_dir: str, chunk_size: int) -> None:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"POG570 file not found: {input_path}")

    os.makedirs(output_dir, exist_ok=True)
    sample_rows: dict[str, list[dict]] = {}

    print(f"\n[POG570] Reading {input_path}...")
    reader = pd.read_csv(
        input_path,
        sep="\t",
        usecols=["chrom", "pos", "ref", "alt", "patient_id", "gene_id"],
        dtype=str,
        chunksize=chunk_size,
    )
    for chunk in tqdm(reader, desc="POG570 chunks"):
        chunk = chunk.dropna(subset=["patient_id", "chrom", "pos", "ref", "alt"])
        for _, row in chunk.iterrows():
            sample_id = str(row["patient_id"]).strip()
            mutation  = f"{row['chrom'].strip()}:{row['pos'].strip()}:{row['ref'].strip()}:{row['alt'].strip()}"
            gene      = str(row["gene_id"]).strip()
            if sample_id not in sample_rows:
                sample_rows[sample_id] = []
            sample_rows[sample_id].append({"mutation": mutation, "gene": gene})

    print(f"[POG570] Writing {len(sample_rows):,} sample files to {output_dir}/...")
    write_sample_files(sample_rows, output_dir)
    print(f"[POG570] Done.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    process_pcawg(args.pcawg_dir, args.pcawg_out, args.chunk_size)
    process_pog570(args.pog570,   args.pog570_out, args.chunk_size)
    print("\nAll done.")


if __name__ == "__main__":
    main()
