#!/usr/bin/env python
"""Get AlphaGenome variant effect scores for each mutation in a patient.

Scores are computed using the CENTER_MASK scorer (DIFF_LOG2_SUM) for output
track predictions (ATAC, CAGE, PROCAP, DNASE, CHIP_TF, CHIP_HISTONE), restricted
to human tracks corresponding to the patient's cancer type tissue.

Tissue specificity is determined by matching the cancer type's 'keyword' column
from metadata/tcga_cancer_tissue_cell_lines.tsv against the 'biosample_name'
column in metadata/alphagenome_metadata.csv (case-insensitive substring match).

Usage:
    cancer-model python scripts/get_alphagenome_mutation_scores.py TCGA-02-0016 \\
        --api-key YOUR_API_KEY \\
        [--server GRPC_ADDRESS] \\
        [--output-dir data/alphagenome_scores] \\
        [--cancer-type GBM] \\
        [--batch-size 1000] \\
        [--max-workers 8] \\
        [--max-mutations N]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from alphagenome.data import genome
from alphagenome.models import dna_client, dna_output, variant_scorers as vs_lib
from alphagenome.models.dna_model import Organism

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).parent.parent
PATIENT_JSON_DIR = PROJECT_DIR / "data" / "patient_json"
PATIENT_VARIANTS_DIR = PROJECT_DIR / "data" / "TCGA" / "tcga_patient_variants"
TISSUE_METADATA = PROJECT_DIR / "metadata" / "tcga_cancer_tissue_cell_lines.tsv"
ALPHAGENOME_METADATA = PROJECT_DIR / "metadata" / "alphagenome_metadata.csv"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data" / "alphagenome_scores"
DEFAULT_DRIVER_GENE_FILE = PROJECT_DIR / "data" / "driver_genes_coords" / "Pancancer.tsv"

# Sequence length for the interval centered on each variant (100 KB = 131 072 bp)
SEQUENCE_LENGTH = dna_client.SEQUENCE_LENGTH_100KB

# ---------------------------------------------------------------------------
# Variant scorers – CENTER_MASK with DIFF_LOG2_SUM (recommended settings)
# ---------------------------------------------------------------------------
OUTPUT_TYPE_SCORERS = {
    "ATAC": vs_lib.CenterMaskScorer(
        requested_output=dna_output.OutputType.ATAC,
        width=501,
        aggregation_type=vs_lib.AggregationType.DIFF_LOG2_SUM,
    ),
    "DNASE": vs_lib.CenterMaskScorer(
        requested_output=dna_output.OutputType.DNASE,
        width=501,
        aggregation_type=vs_lib.AggregationType.DIFF_LOG2_SUM,
    ),
    "CAGE": vs_lib.CenterMaskScorer(
        requested_output=dna_output.OutputType.CAGE,
        width=501,
        aggregation_type=vs_lib.AggregationType.DIFF_LOG2_SUM,
    ),
    "PROCAP": vs_lib.CenterMaskScorer(
        requested_output=dna_output.OutputType.PROCAP,
        width=501,
        aggregation_type=vs_lib.AggregationType.DIFF_LOG2_SUM,
    ),
    "CHIP_TF": vs_lib.CenterMaskScorer(
        requested_output=dna_output.OutputType.CHIP_TF,
        width=501,
        aggregation_type=vs_lib.AggregationType.DIFF_LOG2_SUM,
    ),
    "CHIP_HISTONE": vs_lib.CenterMaskScorer(
        requested_output=dna_output.OutputType.CHIP_HISTONE,
        width=2001,
        aggregation_type=vs_lib.AggregationType.DIFF_LOG2_SUM,
    ),
}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def parse_mutation(mutation_str: str):
    """Parse 'CHR:START-END:REF:ALT' mutation string.

    The format uses 1-based coordinates; START == END for SNVs.
    Returns (chrom_with_chr_prefix, position_1based, ref_bases, alt_bases).
    """
    parts = mutation_str.split(":")
    if len(parts) != 4:
        raise ValueError(f"Unexpected mutation format: {mutation_str!r}")
    chrom_raw, coord_str, ref, alt = parts
    # Add 'chr' prefix if absent
    chrom = chrom_raw if chrom_raw.startswith("chr") else f"chr{chrom_raw}"
    pos = int(coord_str.split("-")[0])  # 1-based start position
    return chrom, pos, ref, alt


def make_centered_interval(chrom: str, pos: int, ref: str, seq_length: int = SEQUENCE_LENGTH) -> genome.Interval:
    """Return a seq_length-wide Interval centered on the variant (0-based coords)."""
    # AlphaGenome Interval uses 0-based half-open coordinates.
    # The variant's 0-based start is pos-1; center on the midpoint of ref.
    center = (pos - 1) + len(ref) // 2
    start = max(0, center - seq_length // 2)
    end = start + seq_length
    return genome.Interval(chromosome=chrom, start=start, end=end)


def load_patient_cancer_type(patient_id: str) -> str:
    """Load cancer type from patient JSON file."""
    json_path = PATIENT_JSON_DIR / f"{patient_id}.json"
    with open(json_path) as f:
        data = json.load(f)
    cancer_type = data.get("cancer_type")
    if cancer_type is None:
        raise KeyError(f"'cancer_type' not found in {json_path}")
    return cancer_type


def get_tissue_ontology_curieS(cancer_type: str) -> set:
    """Return the set of ontology_curie values matching the cancer type's tissue keywords.

    Matching is performed case-insensitively against biosample_name in the
    AlphaGenome track metadata. Rows with output_type == CONTACT_MAPS are excluded.
    """
    # Load cancer → keyword mapping
    tissue_meta = pd.read_csv(TISSUE_METADATA, sep="\t")
    row = tissue_meta[tissue_meta["cancer_type"] == cancer_type]
    if row.empty:
        raise ValueError(
            f"Cancer type {cancer_type!r} not found in {TISSUE_METADATA}. "
            f"Available: {sorted(tissue_meta['cancer_type'].tolist())}"
        )
    keywords = [kw.strip().lower() for kw in row.iloc[0]["keyword"].split(",") if kw.strip()]

    # Load AlphaGenome track metadata, drop contact-map tracks
    ag_meta = pd.read_csv(ALPHAGENOME_METADATA)
    ag_meta = ag_meta[ag_meta["output_type"] != "OutputType.CONTACT_MAPS"].copy()

    # Case-insensitive keyword match against biosample_name
    def matches(biosample_name) -> bool:
        if pd.isna(biosample_name):
            return False
        name_lower = str(biosample_name).lower()
        return any(kw in name_lower for kw in keywords)

    tissue_tracks = ag_meta[ag_meta["biosample_name"].apply(matches)]
    curieS = set(tissue_tracks["ontology_curie"].dropna().unique().tolist())

    print(
        f"[tissue filter] cancer_type={cancer_type!r}: "
        f"{len(tissue_tracks)} matching tracks, {len(curieS)} unique ontology curieS"
    )
    if not curieS:
        print(
            "  Warning: no tissue-specific ontology curieS found; "
            "all returned tracks will be kept.",
            file=sys.stderr,
        )
    return curieS


def load_mutations(patient_id: str) -> pd.DataFrame:
    """Load mutation TSV for a patient."""
    tsv_path = PATIENT_VARIANTS_DIR / f"{patient_id}.tsv"
    if not tsv_path.exists():
        raise FileNotFoundError(f"Mutation file not found: {tsv_path}")
    df = pd.read_csv(tsv_path, sep="\t")
    return df


def load_driver_genes(driver_gene_file: Path) -> set:
    """Return the set of driver gene names from the Pancancer TSV."""
    df = pd.read_csv(driver_gene_file, sep="\t")
    return set(df["gene"].unique())


def filter_anndata_by_ontology(adata, tissue_curieS: set):
    """Subset AnnData columns to tissue-specific ontology curieS.

    If tissue_curieS is empty (no filter), the original AnnData is returned.
    """
    if not tissue_curieS:
        return adata
    if "ontology_curie" not in adata.var.columns:
        return adata
    mask = adata.var["ontology_curie"].isin(tissue_curieS)
    return adata[:, mask]


def score_mutations_batch(
    client,
    batch_variants: list,
    batch_intervals: list,
    scorer_list: list,
    scorer_names: list,
    tissue_curieS: set,
    mutation_meta: list,
    max_workers: int,
) -> list[dict]:
    """Score a batch of variants and return a list of row dicts (long format).

    Returns one dict per (mutation, scorer, track) combination.
    """
    try:
        all_results = client.score_variants(
            intervals=batch_intervals,
            variants=batch_variants,
            variant_scorers=scorer_list,
            organism=Organism.HOMO_SAPIENS,
            progress_bar=False,
            max_workers=max_workers,
        )
    except Exception as exc:
        print(f"  Error scoring batch: {exc}", file=sys.stderr)
        return []

    rows = []
    for meta, scorer_results in zip(mutation_meta, all_results):
        for scorer_name, adata in zip(scorer_names, scorer_results):
            filtered = filter_anndata_by_ontology(adata, tissue_curieS)
            if filtered.n_vars == 0:
                continue
            scores = np.asarray(filtered.X[0], dtype=np.float32)
            var_df = filtered.var

            for track_idx in range(filtered.n_vars):
                row = dict(meta)
                row["scorer"] = scorer_name
                row["score"] = float(scores[track_idx])
                # Track metadata (may be absent if server doesn't return them)
                for col in ("name", "strand", "ontology_curie", "biosample_name",
                            "biosample_type", "biosample_life_stage"):
                    row[col] = var_df.iloc[track_idx].get(col, np.nan) if col in var_df.columns else np.nan
                rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Get AlphaGenome variant effect scores per mutation for a patient."
    )
    parser.add_argument("patient_id", help="Patient ID, e.g. TCGA-02-0016")
    parser.add_argument(
        "--api-key",
        default=".alphagenome_api_key",
        help="File containing the AlphaGenome API key.",
    )
    parser.add_argument(
        "--server",
        default=None,
        help="gRPC server address (default: dns:///gdmscience.googleapis.com:443).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory to write output parquet files (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--cancer-type",
        default=None,
        help="Override cancer type (otherwise loaded from patient JSON).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of mutations to score per batch (default: 1000).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Max parallel gRPC workers for score_variants (default: 4).",
    )
    parser.add_argument(
        "--max-mutations",
        type=int,
        default=None,
        help="Limit total mutations scored (useful for testing).",
    )
    parser.add_argument(
        "--driver-gene-file",
        default=str(DEFAULT_DRIVER_GENE_FILE),
        help=f"Driver gene coords TSV (default: {DEFAULT_DRIVER_GENE_FILE}).",
    )
    args = parser.parse_args()

    patient_id = args.patient_id
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{patient_id}.csv"

    # Resolve cancer type
    if args.cancer_type:
        cancer_type = args.cancer_type
    else:
        print(f"Loading cancer type for {patient_id}...")
        cancer_type = load_patient_cancer_type(patient_id)
    print(f"Cancer type: {cancer_type}")

    # Get tissue-specific ontology curieS
    tissue_curieS = get_tissue_ontology_curieS(cancer_type)

    # Load driver genes and filter mutations
    driver_gene_file = Path(args.driver_gene_file)
    if not driver_gene_file.exists():
        raise FileNotFoundError(f"Driver gene file not found: {driver_gene_file}")
    driver_genes = load_driver_genes(driver_gene_file)
    print(f"Driver genes loaded: {len(driver_genes)}")

    # Load mutations
    mutations_df = load_mutations(patient_id)
    print(f"Total mutations: {len(mutations_df)}")

    # Keep only mutations whose gene name is a driver gene
    # Gene column format: "GENE_NAME(transcript:change,...)" — extract name before "("
    mutations_df["gene_name"] = mutations_df["gene"].apply(
        lambda x: x.split("(")[0].strip() if isinstance(x, str) else ""
    )
    mutations_df = mutations_df[mutations_df["gene_name"].isin(driver_genes)]
    print(f"Driver-gene mutations: {len(mutations_df)} "
          f"({mutations_df['gene_name'].nunique()} unique driver genes)")

    if mutations_df.empty:
        print(f"No driver-gene mutations found for {patient_id}, exiting.")
        sys.exit(0)

    if args.max_mutations is not None:
        mutations_df = mutations_df.head(args.max_mutations)
    print(f"Processing {len(mutations_df)} mutations for {patient_id}")

    # Build Variant and Interval objects
    variants, intervals, mutation_meta = [], [], []
    n_skipped = 0
    for _, row in mutations_df.iterrows():
        mut_str = row["mutation"]
        try:
            chrom, pos, ref, alt = parse_mutation(mut_str)
            variants.append(
                genome.Variant(
                    chromosome=chrom,
                    position=pos,
                    reference_bases=ref,
                    alternate_bases=alt,
                )
            )
            intervals.append(make_centered_interval(chrom, pos, ref))
            mutation_meta.append(
                {
                    "mutation": mut_str,
                    "chr": chrom,
                    "pos": pos,
                    "ref": ref,
                    "alt": alt,
                    "gene": row.get("gene", ""),
                    "cancer": row.get("cancer", ""),
                    "genic_region": row.get("genic_region", ""),
                    "func_effect": row.get("func_effect", ""),
                }
            )
        except Exception as exc:
            print(f"  Skipping {mut_str!r}: {exc}", file=sys.stderr)
            n_skipped += 1

    if n_skipped:
        print(f"Skipped {n_skipped} unparseable mutations.")

    # Connect to AlphaGenome
    print(f"Connecting to AlphaGenome server...")
    api_key_path = Path(args.api_key)
    if not api_key_path.is_file():
        print(f"ERROR: API key file not found: {api_key_path}", file=sys.stderr)
        sys.exit(1)
    with open(api_key_path, "r") as f:
        api_key = f.read().strip()

    client = dna_client.create(api_key=api_key, address=args.server)

    scorer_names = list(OUTPUT_TYPE_SCORERS.keys())
    scorer_list = list(OUTPUT_TYPE_SCORERS.values())

    # Process in batches with progress reporting
    all_rows = []
    batch_size = args.batch_size
    n_batches = (len(variants) + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        lo = batch_idx * batch_size
        hi = min(lo + batch_size, len(variants))
        print(
            f"Batch {batch_idx + 1}/{n_batches}: mutations {lo + 1}–{hi} "
            f"of {len(variants)}",
            flush=True,
        )
        batch_rows = score_mutations_batch(
            client=client,
            batch_variants=variants[lo:hi],
            batch_intervals=intervals[lo:hi],
            scorer_list=scorer_list,
            scorer_names=scorer_names,
            tissue_curieS=tissue_curieS,
            mutation_meta=mutation_meta[lo:hi],
            max_workers=args.max_workers,
        )
        all_rows.extend(batch_rows)
        print(f"  → {len(batch_rows)} track-score rows accumulated (total: {len(all_rows)})")

    if not all_rows:
        print("No scores produced. Check that tissue keywords match biosample names.", file=sys.stderr)
        sys.exit(1)

    results_df = pd.DataFrame(all_rows)
    # Keep only requested columns
    save_cols = ["mutation", "gene", "scorer", "biosample_name", "ontology_curie"]
    # Ensure columns exist
    existing_cols = [c for c in save_cols if c in results_df.columns]
    saved_df = results_df[existing_cols]
    saved_df.to_csv(output_path, index=False)
    print(f"\nSaved {len(saved_df):,} rows → {output_path}")
    if "mutation" in saved_df.columns:
        print(f"Unique mutations scored: {saved_df['mutation'].nunique():,}")
    if "gene" in saved_df.columns:
        print(f"Unique genes: {saved_df['gene'].nunique():,}")
    if "biosample_name" in saved_df.columns:
        print(f"Unique biosamples: {saved_df['biosample_name'].nunique():,}")


if __name__ == "__main__":
    main()
