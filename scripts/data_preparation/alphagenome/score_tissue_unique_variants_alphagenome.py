#!/usr/bin/env python
"""Get AlphaGenome variant effect scores for organ-unique mutations.

Groups TCGA cancer types by organ (via metadata/cancer_types_acronyms.tsv)
and, for each organ:

  1. Collect variants from all cancer types belonging to that organ.
  2. Deduplicate → one row per unique (mutation, gene_name) pair across the organ.
  3. Save intermediate organ-unique variants TSV to
       data/alphagenome_scores/tissue_unique_variants/{ORGAN}.tsv
     (skipped if file already exists unless --force).
  4. Score the unique variants with the full recommended AlphaGenome variant
     scorer set using tissue-specific tracks matched (union of all cancer types
     in organ) via metadata/tcga_cancer_tissue_cell_lines.tsv.
  5. Save final CSV to data/alphagenome_scores/tissue/{ORGAN}.csv with
     columns include mutation, gene, output_type, scorer, score,
     biosample_name, ontology_curie, and gene/track metadata when available.
     (skipped if file already exists unless --force).

Usage:
    conda run -n cancer-model python scripts/score_tissue_unique_variants_alphagenome.py \\
        [--api-key .alphagenome_api_key] \\
        [--server GRPC_ADDRESS] \\
        [--output-dir data/alphagenome_scores] \\
        [--batch-size 500] \\
        [--max-workers 4] \\
        [--organs CNS Kidney ...]  # optional subset; default: all \\
        [--force]
"""

import argparse
import concurrent.futures
import sys
from pathlib import Path

import pandas as pd

from alphagenome.data import genome
from alphagenome.models import dna_client, variant_scorers as vs_lib
from alphagenome.models.dna_model import Organism
from tqdm.auto import tqdm



# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[3]
CANCER_VARIANTS_DIR = PROJECT_DIR / "data" / "TCGA" / "tcga_patient_variants_by_cancer"
TISSUE_METADATA = PROJECT_DIR / "metadata" / "tcga_cancer_tissue_cell_lines.tsv"
CANCER_ACRONYMS_FILE = PROJECT_DIR / "metadata" / "cancer_types_acronyms.tsv"
ALPHAGENOME_METADATA = PROJECT_DIR / "metadata" / "alphagenome_metadata.csv"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data" / "alphagenome_scores"
DEFAULT_API_KEY = ".alphagenome_api_key"
DEFAULT_SERVER = None
DEFAULT_SEQUENCE_LENGTH_NAME = "1mb"
DEFAULT_SCORERS = None
DEFAULT_BATCH_SIZE = 500
DEFAULT_MAX_WORKERS = 1
DEFAULT_SHOW_PROGRESS = True
DEFAULT_ORGANS = None
DEFAULT_FORCE = False
DEFAULT_DRIVER_GENE_FILE = None  # Deprecated no-op; retained for old commands.

UNIQUE_VARIANTS_DIR_NAME = "tissue_unique_variants"
TISSUE_SCORES_DIR_NAME = "tissue"
COMPLETED_MUTATIONS_SUFFIX = ".completed_mutations.tsv"

SEQUENCE_LENGTHS = {
    "16kb": dna_client.SEQUENCE_LENGTH_16KB,
    "100kb": dna_client.SEQUENCE_LENGTH_100KB,
    "500kb": dna_client.SEQUENCE_LENGTH_500KB,
    "1mb": dna_client.SEQUENCE_LENGTH_1MB,
}
SEQUENCE_LENGTH = SEQUENCE_LENGTHS[DEFAULT_SEQUENCE_LENGTH_NAME]
PREFERRED_SCORE_COLUMNS = [
    "mutation", "gene", "output_type", "scorer", "score", "quantile_score",
    "biosample_name", "ontology_curie", "track_name", "track_strand",
    "Assay title", "biosample_type", "biosample_life_stage", "data_source",
    "gtex_tissue", "transcription_factor", "histone_mark", "gene_id",
    "gene_name", "gene_type", "gene_strand", "junction_Start",
    "junction_End", "scored_interval", "variant_id", "variant_scorer",
    "raw_score",
]

# ---------------------------------------------------------------------------
# Step 1 – Aggregation helpers
# ---------------------------------------------------------------------------


def sanitize_organ_name(organ: str) -> str:
    """Convert organ name to a safe filename (spaces → underscores)."""
    return organ.replace(" ", "_")


def build_organ_cancer_map(available_cancer_types: set) -> dict:
    """Return {organ: [acronyms]} filtered to cancer types present in data.

    Reads metadata/cancer_types_acronyms.tsv, groups Acronym by Organ,
    and keeps only acronyms that exist in available_cancer_types.
    Organs with no available acronyms are dropped.
    """
    df = pd.read_csv(CANCER_ACRONYMS_FILE, sep="\t")
    organ_map: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        acronym = str(row["Acronym"]).strip()
        organ = str(row["Organ"]).strip()
        if pd.isna(row["Organ"]) or pd.isna(row["Acronym"]):
            continue
        if acronym not in available_cancer_types:
            continue
        organ_map.setdefault(organ, []).append(acronym)
    return organ_map


def build_organ_unique_variants(
    organ: str,
    cancer_type_list: list,
    cancer_variants_dir: Path,
) -> pd.DataFrame:
    """Collect and deduplicate variants across all cancer types in an organ.

    Reads {CT}_variants.tsv for each cancer type, concatenates, and deduplicates
    on (mutation, gene_name) across the organ.
    Returns DataFrame with columns: mutation, gene.
    """
    parts = []
    for cancer_type in cancer_type_list:
        tsv_path = cancer_variants_dir / f"{cancer_type}_variants.tsv"
        if not tsv_path.exists():
            print(f"  [{cancer_type}] Variant file not found, skipping.", file=sys.stderr)
            continue
        df = pd.read_csv(tsv_path, sep="\t", dtype=str, usecols=["mutation", "gene_name"])
        n_total = len(df)
        print(
            f"  [{cancer_type}] {n_total:,} rows "
            f"({df['gene_name'].nunique()} genes)"
        )
        parts.append(df)

    if not parts:
        return pd.DataFrame(columns=["mutation", "gene"])

    combined = pd.concat(parts, ignore_index=True)
    unique = (
        combined.drop_duplicates(subset=["mutation", "gene_name"])
        .rename(columns={"gene_name": "gene"})
        .reset_index(drop=True)
    )
    print(f"  [{organ}] {len(combined):,} rows → {len(unique):,} unique (mutation, gene) pairs")
    return unique


# ---------------------------------------------------------------------------
# Step 2 – AlphaGenome scoring helpers
# ---------------------------------------------------------------------------


def get_organ_ontology_curies(cancer_type_list: list, organ: str) -> set:
    """Return ontology_curie values for tracks matching the union of tissue keywords
    across all cancer types in the organ.
    """
    tissue_meta = pd.read_csv(TISSUE_METADATA, sep="\t")
    keywords: list[str] = []
    for cancer_type in cancer_type_list:
        row = tissue_meta[tissue_meta["cancer_type"] == cancer_type]
        if row.empty:
            print(f"  Warning: {cancer_type!r} not found in tissue metadata, skipping keywords.", file=sys.stderr)
            continue
        kws = [kw.strip().lower() for kw in str(row.iloc[0]["keyword"]).split(",") if kw.strip()]
        keywords.extend(kws)
    keywords = list(dict.fromkeys(keywords))  # deduplicate, preserve order

    if not keywords:
        print(f"  Warning: no keywords found for organ {organ!r}.", file=sys.stderr)
        return set()

    ag_meta = pd.read_csv(ALPHAGENOME_METADATA)
    ag_meta = ag_meta[ag_meta["output_type"] != "OutputType.CONTACT_MAPS"].copy()

    def matches(biosample_name) -> bool:
        if pd.isna(biosample_name):
            return False
        name_lower = str(biosample_name).lower()
        return any(kw in name_lower for kw in keywords)

    tissue_tracks = ag_meta[ag_meta["biosample_name"].apply(matches)]
    curies = set(tissue_tracks["ontology_curie"].dropna().unique().tolist())
    print(
        f"[tissue filter] organ={organ!r} ({len(cancer_type_list)} cancer types, {len(keywords)} keywords): "
        f"{len(tissue_tracks)} matching tracks, {len(curies)} unique ontology curies"
    )
    if not curies:
        print(
            f"  Warning: no tissue-specific ontology curies found for {organ!r}; "
            "all returned tracks will be kept.",
            file=sys.stderr,
        )
    return curies


def parse_mutation(mutation_str: str):
    """Parse 'CHR:START-END:REF:ALT' → (chr_prefixed, pos_1based, ref, alt)."""
    parts = mutation_str.split(":")
    if len(parts) != 4:
        raise ValueError(f"Unexpected mutation format: {mutation_str!r}")
    chrom_raw, coord_str, ref, alt = parts
    chrom = chrom_raw if chrom_raw.startswith("chr") else f"chr{chrom_raw}"
    pos = int(coord_str.split("-")[0])
    return chrom, pos, ref, alt


def make_centered_interval(chrom: str, pos: int, ref: str, seq_length: int) -> genome.Interval:
    """Return seq_length-wide Interval (0-based) centered on the variant."""
    center = (pos - 1) + len(ref) // 2
    start = max(0, center - seq_length // 2)
    end = start + seq_length
    return genome.Interval(chromosome=chrom, start=start, end=end)


def filter_anndata_by_ontology(adata, curies: set):
    """Subset AnnData columns to tissue-specific ontology curies."""
    if not curies or "ontology_curie" not in adata.var.columns:
        return adata
    mask = adata.var["ontology_curie"].isin(curies)
    return adata[:, mask]


def recommended_scorers_by_name() -> dict[str, object]:
    """Return AlphaGenome's recommended variant scorers keyed by stable names."""
    if hasattr(vs_lib, "RECOMMENDED_VARIANT_SCORERS"):
        return {str(name).upper(): scorer for name, scorer in dict(vs_lib.RECOMMENDED_VARIANT_SCORERS).items()}
    return {
        str(getattr(scorer, "name", "") or scorer.__class__.__name__).upper(): scorer
        for scorer in vs_lib.get_recommended_scorers(Organism.HOMO_SAPIENS.to_proto())
    }


def select_scorers(requested_names: list[str] | None) -> tuple[list[str], list[object]]:
    """Select recommended scorers, preserving AlphaGenome's recommended order."""
    scorer_map = recommended_scorers_by_name()
    if not requested_names:
        return list(scorer_map.keys()), list(scorer_map.values())

    requested = [name.upper() for name in requested_names]
    missing = [name for name in requested if name not in scorer_map]
    if missing:
        raise ValueError(f"Unknown recommended scorer(s): {missing}. Available: {sorted(scorer_map)}")
    return requested, [scorer_map[name] for name in requested]


def score_one_variant(client, interval, variant, scorer_list):
    """Score one variant using the single-variant API shared by local/remote clients."""
    return client.score_variant(
        interval=interval,
        variant=variant,
        variant_scorers=scorer_list,
        organism=Organism.HOMO_SAPIENS,
    )


def tidy_variant_result(meta: dict, scorer_results: list, tissue_curies: set) -> list[dict]:
    """Convert AlphaGenome AnnData scorer outputs to long-format rows."""
    filtered_results = []
    for adata in scorer_results:
        filtered = filter_anndata_by_ontology(adata, tissue_curies)
        if filtered.n_obs > 0 and filtered.n_vars > 0:
            filtered_results.append(filtered)
    if not filtered_results:
        return []

    tidy_df = vs_lib.tidy_scores([filtered_results])
    if tidy_df is None or tidy_df.empty:
        return []

    tidy_df.insert(0, "gene", meta.get("gene", ""))
    tidy_df.insert(0, "mutation", meta.get("mutation", ""))
    if "variant_scorer" in tidy_df.columns:
        tidy_df["scorer"] = tidy_df["variant_scorer"]
    if "raw_score" in tidy_df.columns:
        tidy_df["score"] = tidy_df["raw_score"]
    return tidy_df.to_dict(orient="records")


def progress_bar(iterable, *, total: int, desc: str, enabled: bool):
    if not enabled or tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit="variant", leave=False)


def score_variants_batch(
    client,
    batch_variants: list,
    batch_intervals: list,
    scorer_list: list,
    tissue_curies: set,
    mutation_meta: list,
    max_workers: int,
    show_progress: bool,
    progress_label: str,
) -> tuple[list[dict], list[str]]:
    """Score a batch and return long-format row dicts."""
    rows = []
    completed_mutations = []
    jobs = list(zip(mutation_meta, batch_intervals, batch_variants, strict=True))
    if max_workers <= 1:
        for meta, interval, variant in progress_bar(
            jobs,
            total=len(jobs),
            desc=progress_label,
            enabled=show_progress,
        ):
            try:
                scorer_results = score_one_variant(client, interval, variant, scorer_list)
            except Exception as exc:
                print(f"  Error scoring {meta.get('mutation', '<unknown>')}: {exc}", file=sys.stderr)
                continue
            rows.extend(tidy_variant_result(meta, scorer_results, tissue_curies))
            completed_mutations.append(meta["mutation"])
        return rows, completed_mutations

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_meta = {
            executor.submit(score_one_variant, client, interval, variant, scorer_list): meta
            for meta, interval, variant in jobs
        }
        completed = concurrent.futures.as_completed(future_to_meta)
        for future in progress_bar(
            completed,
            total=len(future_to_meta),
            desc=progress_label,
            enabled=show_progress,
        ):
            meta = future_to_meta[future]
            try:
                scorer_results = future.result()
            except Exception as exc:
                print(f"  Error scoring {meta.get('mutation', '<unknown>')}: {exc}", file=sys.stderr)
                continue
            rows.extend(tidy_variant_result(meta, scorer_results, tissue_curies))
            completed_mutations.append(meta["mutation"])
    return rows, completed_mutations


def load_completed_mutations(completed_path: Path, scored_path: Path) -> set[str]:
    """Load completed mutations, falling back to existing score rows if needed."""
    completed = set()
    if completed_path.exists():
        completed_df = pd.read_csv(completed_path, sep="\t", dtype=str)
        if "mutation" in completed_df.columns:
            completed.update(completed_df["mutation"].dropna())

    if scored_path.exists():
        try:
            scored_df = pd.read_csv(scored_path, usecols=["mutation"], dtype=str)
        except (ValueError, pd.errors.EmptyDataError):
            return completed
        completed.update(scored_df["mutation"].dropna())

    return completed


def append_completed_mutations(completed_path: Path, mutations: list[str]) -> None:
    if not mutations:
        return
    completed_df = pd.DataFrame({"mutation": mutations})
    completed_df.to_csv(
        completed_path,
        sep="\t",
        index=False,
        mode="a",
        header=not completed_path.exists() or completed_path.stat().st_size == 0,
    )


def append_score_rows(scored_path: Path, rows: list[dict]) -> int:
    if not rows:
        return 0
    results_df = pd.DataFrame(rows)
    save_cols = [c for c in PREFERRED_SCORE_COLUMNS if c in results_df.columns]
    save_cols.extend(c for c in results_df.columns if c not in save_cols)
    results_df[save_cols].to_csv(
        scored_path,
        index=False,
        mode="a",
        header=not scored_path.exists() or scored_path.stat().st_size == 0,
    )
    return len(results_df)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "For every organ, aggregate variants from all its TCGA "
            "cancer types and score them with AlphaGenome."
        )
    )
    parser.add_argument(
        "--api-key",
        default=DEFAULT_API_KEY,
        help="File containing the AlphaGenome API key (default: %(default)s).",
    )
    parser.add_argument(
        "--server",
        default=DEFAULT_SERVER,
        help="gRPC server address (default: dns:///gdmscience.googleapis.com:443).",
    )
    parser.add_argument(
        "--driver-gene-file",
        default=DEFAULT_DRIVER_GENE_FILE,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--sequence-length",
        choices=sorted(SEQUENCE_LENGTHS),
        default=DEFAULT_SEQUENCE_LENGTH_NAME,
        help="Centered AlphaGenome interval length (default: %(default)s). Use 1mb for contact maps.",
    )
    parser.add_argument(
        "--scorers",
        nargs="+",
        default=DEFAULT_SCORERS,
        metavar="SCORER",
        help="Optional subset of recommended scorers to run (default: all recommended scorers).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Root output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Mutations per scoring batch/chunk (default: %(default)s).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help="Max parallel gRPC workers (default: %(default)s).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_false",
        dest="show_progress",
        default=DEFAULT_SHOW_PROGRESS,
        help="Disable per-batch scoring progress bars.",
    )
    parser.add_argument(
        "--organs",
        nargs="+",
        default=DEFAULT_ORGANS,
        metavar="ORGAN",
        help="Optional subset of organs to process (default: all). "
             "Use organ names as in metadata/cancer_types_acronyms.tsv, e.g. CNS Kidney Lung.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=DEFAULT_FORCE,
        help="Recompute files even if they already exist.",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be a positive integer.")
    if args.max_workers < 1:
        parser.error("--max-workers must be a positive integer.")
    args.sequence_length = SEQUENCE_LENGTHS[args.sequence_length]

    output_dir = Path(args.output_dir)
    unique_variants_dir = output_dir / UNIQUE_VARIANTS_DIR_NAME
    tissue_scores_dir = output_dir / TISSUE_SCORES_DIR_NAME
    unique_variants_dir.mkdir(parents=True, exist_ok=True)
    tissue_scores_dir.mkdir(parents=True, exist_ok=True)

    # Build organ → cancer type list mapping (filtered to data actually available)
    available_cancer_types = set(
        p.name.replace("_variants.tsv", "")
        for p in CANCER_VARIANTS_DIR.glob("*_variants.tsv")
    )
    if not available_cancer_types:
        print(f"No *_variants.tsv files found in {CANCER_VARIANTS_DIR}", file=sys.stderr)
        sys.exit(1)

    organ_map = build_organ_cancer_map(available_cancer_types)
    if not organ_map:
        print("No organs found in cancer_types_acronyms.tsv matching available data.", file=sys.stderr)
        sys.exit(1)

    # Optionally filter to requested organs
    if args.organs:
        unknown = [o for o in args.organs if o not in organ_map]
        if unknown:
            print(f"Warning: organs not found in mapping: {unknown}", file=sys.stderr)
        organs_to_process = [o for o in args.organs if o in organ_map]
    else:
        organs_to_process = sorted(organ_map.keys())

    print(f"\nOrgans to process ({len(organs_to_process)}):")
    for organ in organs_to_process:
        print(f"  {organ}: {organ_map[organ]}")
    print()

    # ------------------------------------------------------------------
    # Phase 1: Build organ-unique variant files
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Phase 1: Building organ-unique variant files")
    print("=" * 60)
    for organ in organs_to_process:
        organ_fname = sanitize_organ_name(organ)
        unique_path = unique_variants_dir / f"{organ_fname}.tsv"
        if unique_path.exists() and not args.force:
            n = sum(1 for _ in open(unique_path)) - 1
            print(f"\n[{organ}] Skipping (exists, {n:,} rows): {unique_path}")
            continue
        print(f"\n[{organ}] Building organ-unique variants from {organ_map[organ]} ...")
        try:
            unique_df = build_organ_unique_variants(
                organ, organ_map[organ], CANCER_VARIANTS_DIR
            )
            unique_df.to_csv(unique_path, sep="\t", index=False)
            print(f"  Saved → {unique_path}")
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Phase 2: Score each organ with AlphaGenome
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Phase 2: AlphaGenome scoring")
    print("=" * 60)

    api_key_path = Path(args.api_key)
    if not api_key_path.is_file():
        print(f"ERROR: API key file not found: {api_key_path}", file=sys.stderr)
        sys.exit(1)
    with open(api_key_path) as f:
        api_key = f.read().strip()

    print("Connecting to AlphaGenome server ...")
    client = dna_client.create(api_key=api_key, address=args.server)

    scorer_names, scorer_list = select_scorers(args.scorers)
    print(f"Recommended scorers to run ({len(scorer_names)}): {', '.join(scorer_names)}")

    for organ in organs_to_process:
        organ_fname = sanitize_organ_name(organ)
        scored_path = tissue_scores_dir / f"{organ_fname}.csv"
        completed_path = tissue_scores_dir / f"{organ_fname}{COMPLETED_MUTATIONS_SUFFIX}"

        if args.force:
            for path in (scored_path, completed_path):
                if path.exists():
                    path.unlink()

        unique_path = unique_variants_dir / f"{organ_fname}.tsv"
        if not unique_path.exists():
            print(f"\n[{organ}] No unique-variants file found, skipping.", file=sys.stderr)
            continue

        print(f"\n[{organ}] Scoring ({organ_map[organ]}) ...")
        unique_df = pd.read_csv(unique_path, sep="\t", dtype=str)
        print(f"  {len(unique_df):,} unique mutations to score")

        # Organ-wide tissue-specific track filter (union of all cancer types)
        tissue_curies = get_organ_ontology_curies(organ_map[organ], organ)

        # Build Variant + Interval objects
        variants, intervals, mutation_meta = [], [], []
        n_skipped = 0
        for _, row in unique_df.iterrows():
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
                intervals.append(make_centered_interval(chrom, pos, ref, args.sequence_length))
                mutation_meta.append({"mutation": mut_str, "gene": row.get("gene", "")})
            except Exception as exc:
                print(f"  Skipping {mut_str!r}: {exc}", file=sys.stderr)
                n_skipped += 1

        if n_skipped:
            print(f"  Skipped {n_skipped} unparseable mutations.")
        if not variants:
            print(f"  No valid variants for {organ}, skipping.")
            continue

        completed_mutations = load_completed_mutations(completed_path, scored_path)
        if completed_mutations:
            keep_indices = [
                idx for idx, meta in enumerate(mutation_meta)
                if meta["mutation"] not in completed_mutations
            ]
            n_completed = len(variants) - len(keep_indices)
            variants = [variants[idx] for idx in keep_indices]
            intervals = [intervals[idx] for idx in keep_indices]
            mutation_meta = [mutation_meta[idx] for idx in keep_indices]
            print(
                f"  Resume state: {n_completed:,} completed, "
                f"{len(variants):,} remaining"
            )

        if not variants:
            print(f"  All valid variants already completed for {organ}, skipping.")
            continue

        # Score in batches
        total_rows_written = 0
        batch_size = args.batch_size
        n_batches = (len(variants) + batch_size - 1) // batch_size
        for batch_idx in range(n_batches):
            lo = batch_idx * batch_size
            hi = min(lo + batch_size, len(variants))
            print(
                f"  Batch {batch_idx + 1}/{n_batches}: "
                f"mutations {lo + 1}–{hi} of {len(variants)}",
                flush=True,
            )
            batch_rows, batch_completed = score_variants_batch(
                client=client,
                batch_variants=variants[lo:hi],
                batch_intervals=intervals[lo:hi],
                scorer_list=scorer_list,
                tissue_curies=tissue_curies,
                mutation_meta=mutation_meta[lo:hi],
                max_workers=args.max_workers,
                show_progress=args.show_progress,
                progress_label=f"{organ} batch {batch_idx + 1}/{n_batches}",
            )
            rows_written = append_score_rows(scored_path, batch_rows)
            append_completed_mutations(completed_path, batch_completed)
            total_rows_written += rows_written
            print(
                f"    → {rows_written:,} rows saved, "
                f"{len(batch_completed):,} mutations completed "
                f"(session total rows: {total_rows_written:,})"
            )

        print(f"  Saved/resumed scores → {scored_path}")
        print(f"  Completion ledger → {completed_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
