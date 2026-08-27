#!/usr/bin/env python3
"""Score unique variants in the top cancer-specific driver genes with AlphaGenome.

Example:
    python scripts/data_preparation/score_cancer_specific_driver_variants_alphagenome.py BRCA
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from alphagenome.data import genome
from alphagenome.models import dna_client, variant_scorers
from alphagenome.models.dna_model import Organism


REPO_ROOT = Path(__file__).resolve().parents[2]
SEQUENCE_LENGTH = dna_client.SEQUENCE_LENGTH_100KB
SCORER_NAMES = [
    "RNA_SEQ", "CAGE", "PROCAP", "ATAC", "DNASE", "CHIP_HISTONE", "CHIP_TF",
    "POLYADENYLATION", "SPLICE_SITES", "SPLICE_SITE_USAGE", "SPLICE_JUNCTIONS",
]
SCORERS = [
    (name, variant_scorers.RECOMMENDED_VARIANT_SCORERS[name]) for name in SCORER_NAMES
]


def parse_variant(mutation):
    chrom, coordinates, reference, alternate = mutation.split(":")
    return genome.Variant(
        chromosome=chrom if chrom.startswith("chr") else f"chr{chrom}",
        position=int(coordinates.split("-")[0]),
        reference_bases=reference.replace("-", ""),
        alternate_bases=alternate.replace("-", ""),
        name=mutation,
    )


def score_batch(client, variants, workers):
    results = client.score_variants(
        intervals=[variant.reference_interval.resize(SEQUENCE_LENGTH) for variant in variants],
        variants=variants,
        variant_scorers=[scorer for _, scorer in SCORERS],
        organism=Organism.HOMO_SAPIENS,
        progress_bar=False,
        max_workers=workers,
    )
    rows = []
    for variant, scorer_results in zip(variants, results):
        for (output_type, _), result in zip(SCORERS, scorer_results):
            scores = np.ravel(np.asarray(result.X, dtype=float))
            scores = scores[np.isfinite(scores)]
            rows.append({
                "mutation": variant.name,
                "output_type": output_type,
                "predicted_effect": float(np.mean(scores)) if len(scores) else np.nan,
                "predicted_absolute_effect": float(np.mean(np.abs(scores))) if len(scores) else np.nan,
                "max_absolute_effect": float(np.max(np.abs(scores))) if len(scores) else np.nan,
                "n_tracks": len(scores),
            })
    return rows


def load_unique_variants(variants_file, genes, chunksize):
    variant_genes = defaultdict(set)
    variant_patients = defaultdict(set)
    for chunk in tqdm(
        pd.read_csv(
            variants_file,
            sep="\t",
            usecols=["bcr_patient_barcode", "mutation", "gene_name"],
            dtype=str,
            chunksize=chunksize,
        ),
        desc="Filtering cancer variants",
    ):
        chunk = chunk.loc[chunk["gene_name"].isin(genes)]
        for mutation, rows in chunk.groupby("mutation"):
            variant_genes[mutation].update(rows["gene_name"])
            variant_patients[mutation].update(rows["bcr_patient_barcode"])

    if not variant_genes:
        raise ValueError(f"No variants in the selected driver genes were found in {variants_file}")
    return pd.DataFrame(
        {
            "mutation": mutation,
            "genes": ",".join(sorted(variant_genes[mutation])),
            "n_patients": len(variant_patients[mutation]),
        }
        for mutation in sorted(variant_genes)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cancer_type", help="TCGA cancer code, e.g. BRCA")
    parser.add_argument("--top-genes", type=int, default=10)
    parser.add_argument("--driver-genes-dir", type=Path, default=REPO_ROOT / "data/driver_genes_coords")
    parser.add_argument("--variants-dir", type=Path, default=REPO_ROOT / "data/TCGA/tcga_patient_variants_by_cancer")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--api-key-file", type=Path, default=REPO_ROOT / ".alphagenome_api_key")
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    cancer_type = args.cancer_type.upper()
    driver_file = args.driver_genes_dir / f"{cancer_type}.tsv"
    variants_file = args.variants_dir / f"{cancer_type}_variants.tsv"
    driver_genes = pd.read_csv(driver_file, sep="\t", usecols=["gene"])["gene"].dropna().head(args.top_genes)
    if len(driver_genes) < args.top_genes:
        raise ValueError(f"{driver_file} contains only {len(driver_genes)} driver genes")

    unique_variants = load_unique_variants(variants_file, set(driver_genes), args.chunk_size)
    output = args.output or REPO_ROOT / "data/alphagenome_scores" / f"{cancer_type}_top{args.top_genes}_driver_variant_scores.tsv"
    print(f"Scoring {len(unique_variants):,} unique variants in: {', '.join(driver_genes)}")

    client = dna_client.create(api_key=args.api_key_file.read_text().strip())
    score_rows = []
    mutations = unique_variants["mutation"].tolist()
    for start in tqdm(range(0, len(mutations), args.batch_size), desc="Scoring variants"):
        variants = [parse_variant(mutation) for mutation in mutations[start:start + args.batch_size]]
        score_rows.extend(score_batch(client, variants, args.workers))

    scores = pd.DataFrame(score_rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    unique_variants.merge(scores, on="mutation", how="left").to_csv(output, sep="\t", index=False)
    print(f"Saved {len(scores):,} AlphaGenome scores to {output}")


if __name__ == "__main__":
    main()
