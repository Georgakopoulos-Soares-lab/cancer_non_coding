#!/usr/bin/env python
"""Get local AlphaGenome variant effect scores for organ-unique mutations.

Groups TCGA cancer types by organ (via metadata/cancer_types_acronyms.tsv)
and, for each organ:

  1. Collect variants from all cancer types belonging to that organ.
  2. Deduplicate → one row per unique (mutation, gene_name) pair across the organ.
  3. Save intermediate organ-unique variants TSV to
       data/alphagenome_scores/tissue_unique_variants/{ORGAN}.tsv
     (skipped if file already exists unless --force).
  4. Score the unique variants with the full recommended AlphaGenome variant
     scorer set and tissue-specific tracks matched (union of all cancer types
     in organ) via metadata/tcga_cancer_tissue_cell_lines.tsv.
  5. Save final CSV to data/alphagenome_scores/tissue_local/{ORGAN}.csv with
     columns include mutation, gene, output_type, scorer, score,
     biosample_name, ontology_curie, and gene/track metadata when available.
     (skipped if file already exists unless --force).

Usage:
    conda run -n cancer-model python scripts/data_preparation/alphagenome/score_tissue_unique_variants_alphagenome_local.py \\
        --weights-source huggingface \\
        [--output-dir data/alphagenome_scores] \\
        [--batch-size 500] \\
        [--max-workers 1] \\
        [--organs CNS Kidney ...]  # optional subset; default: all \\
        [--force]

    # Fully offline/local weights and references:
    conda run -n cancer-model python scripts/data_preparation/alphagenome/score_tissue_unique_variants_alphagenome_local.py \\
        --weights-source checkpoint \\
        --checkpoint-path /path/to/alphagenome-all-folds \\
        --fasta-path /path/to/GRCh38.p13.genome.fa \\
        --gtf-feather-path /path/to/gencode.v46.annotation.gtf.gz.feather \\
        --pas-feather-path /path/to/polyadb_human_v3_exon3_contiguous_gtfv46.feather
"""

import argparse
import concurrent.futures
import importlib
import inspect
import sys
import time
from pathlib import Path

import pandas as pd

from alphagenome.data import genome
from alphagenome.models import dna_client, variant_scorers as vs_lib
from alphagenome.models.dna_model import Organism

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[3]
CANCER_VARIANTS_DIR = PROJECT_DIR / "data" / "TCGA" / "tcga_patient_variants_by_cancer"
TISSUE_METADATA = PROJECT_DIR / "metadata" / "tcga_cancer_tissue_cell_lines.tsv"
CANCER_ACRONYMS_FILE = PROJECT_DIR / "metadata" / "cancer_types_acronyms.tsv"
ALPHAGENOME_METADATA = PROJECT_DIR / "metadata" / "alphagenome_metadata.csv"
ALPHAGENOME_RESEARCH_SRC = PROJECT_DIR / "alphagenome_research" / "src"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data" / "alphagenome_scores"
DEFAULT_WEIGHTS_SOURCE = "huggingface"
DEFAULT_CHECKPOINT_PATH = None
DEFAULT_MODEL_VERSION = "all_folds"
DEFAULT_DEVICE = "auto"
DEFAULT_ALLOW_CPU = False
DEFAULT_FASTA_PATH = None
DEFAULT_GTF_FEATHER_PATH = None
DEFAULT_PAS_FEATHER_PATH = None
DEFAULT_SPLICE_SITE_STARTS_FEATHER_PATH = None
DEFAULT_SPLICE_SITE_ENDS_FEATHER_PATH = None
DEFAULT_CALIBRATION_PATH = None
DEFAULT_SEQUENCE_LENGTH_NAME = "1mb"
DEFAULT_SCORERS = None
DEFAULT_BATCH_SIZE = 500
DEFAULT_MAX_WORKERS = "auto"
DEFAULT_SHOW_PROGRESS = True
DEFAULT_VERBOSE_MODEL_LOADING = True
DEFAULT_ORGANS = None
DEFAULT_FORCE = False
DEFAULT_DRIVER_GENE_FILE = None  # Deprecated no-op; retained for old commands.

UNIQUE_VARIANTS_DIR_NAME = "tissue_unique_variants"
TISSUE_SCORES_DIR_NAME = "tissue_local"
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


def log_model_loading(message: str, enabled: bool = True) -> None:
    if enabled:
        print(f"[model load {time.strftime('%H:%M:%S')}] {message}", flush=True)


def resolve_max_workers(value: str) -> int:
    """Resolve max workers from an integer string or visible GPU device count."""
    if str(value).lower() != "auto":
        try:
            workers = int(value)
        except ValueError as exc:
            raise ValueError("--max-workers must be a positive integer or 'auto'.") from exc
        if workers < 1:
            raise ValueError("--max-workers must be a positive integer or 'auto'.")
        return workers

    jax = importlib.import_module("jax")
    gpu_devices = [device for device in jax.local_devices() if device.platform == "gpu"]
    workers = max(1, len(gpu_devices))
    print(f"Auto max workers: {workers} ({len(gpu_devices)} visible GPU devices)")
    return workers


def resolve_jax_device(device_name: str | None, allow_cpu: bool = False):
    """Return a JAX device for local AlphaGenome, or None for its default."""
    if device_name in {None, "", "auto"}:
        return None

    jax = importlib.import_module("jax")
    if device_name in {"cpu", "gpu", "tpu"}:
        devices = jax.devices(device_name)
        if not devices:
            raise ValueError(f"No JAX devices found for platform {device_name!r}.")
        device = devices[0]
    else:
        matches = [device for device in jax.local_devices() if str(device) == device_name]
        if not matches:
            available = ", ".join(str(device) for device in jax.local_devices())
            raise ValueError(f"JAX device {device_name!r} not found. Available: {available}")
        device = matches[0]

    if device.platform == "cpu" and not allow_cpu:
        raise ValueError("Refusing to run AlphaGenome on CPU; pass --allow-cpu to opt in.")
    return device


def call_with_supported_kwargs(fn, *args, **kwargs):
    """Call a local AlphaGenome factory while tolerating minor signature drift."""
    signature = inspect.signature(fn)
    supported = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters and value is not None
    }
    return fn(*args, **supported)


def build_organism_settings(local_dna_model, args):
    """Build optional local reference-file settings for Homo sapiens."""
    reference_kwargs = {
        "fasta_path": args.fasta_path,
        "gtf_feather_path": args.gtf_feather_path,
        "pas_feather_path": args.pas_feather_path,
        "splice_site_starts_feather_path": args.splice_site_starts_feather_path,
        "splice_site_ends_feather_path": args.splice_site_ends_feather_path,
        "calibration_path": args.calibration_path,
    }
    reference_kwargs = {
        key: str(Path(value))
        for key, value in reference_kwargs.items()
        if value is not None
    }
    if not reference_kwargs:
        return None

    if not hasattr(local_dna_model, "OrganismSettings"):
        raise ImportError(
            "alphagenome_research.model.dna_model has no OrganismSettings; "
            "update alphagenome_research or use --weights-source huggingface/kaggle."
        )

    settings_signature = inspect.signature(local_dna_model.OrganismSettings)
    supported_kwargs = {
        key: value
        for key, value in reference_kwargs.items()
        if key in settings_signature.parameters
    }
    ignored = sorted(set(reference_kwargs) - set(supported_kwargs))
    if ignored:
        print(f"Warning: local OrganismSettings does not support {ignored}; ignoring.", file=sys.stderr)

    organism_enum = getattr(local_dna_model, "Organism", Organism)
    if hasattr(local_dna_model, "default_organism_settings"):
        default_settings = local_dna_model.default_organism_settings().get(
            organism_enum.HOMO_SAPIENS
        )
        if default_settings is not None:
            merged_kwargs = {
                key: getattr(default_settings, key)
                for key in settings_signature.parameters
                if hasattr(default_settings, key)
            }
            merged_kwargs.update(supported_kwargs)
            return {
                organism_enum.HOMO_SAPIENS: local_dna_model.OrganismSettings(
                    **merged_kwargs
                )
            }

    return {
        organism_enum.HOMO_SAPIENS: local_dna_model.OrganismSettings(
            **supported_kwargs
        )
    }


def create_local_model(args):
    """Create an AlphaGenome Research model for local scoring."""
    load_started = time.monotonic()
    log_model_loading("Starting local AlphaGenome model setup.", args.verbose_model_loading)

    if ALPHAGENOME_RESEARCH_SRC.exists() and str(ALPHAGENOME_RESEARCH_SRC) not in sys.path:
        sys.path.insert(0, str(ALPHAGENOME_RESEARCH_SRC))
        log_model_loading(
            f"Added AlphaGenome Research source path: {ALPHAGENOME_RESEARCH_SRC}",
            args.verbose_model_loading,
        )

    log_model_loading("Importing alphagenome_research.model.dna_model ...", args.verbose_model_loading)
    try:
        from alphagenome_research.model import dna_model as local_dna_model
    except ModuleNotFoundError as exc:
        if exc.name != "alphagenome_research":
            raise
        raise ModuleNotFoundError(
            "Local scoring requires the AlphaGenome Research package. Install it "
            "from https://github.com/google-deepmind/alphagenome_research "
            "(for example: pip install -e ./alphagenome_research), or place it "
            f"at {ALPHAGENOME_RESEARCH_SRC}."
        ) from exc
    log_model_loading("Imported alphagenome_research.model.dna_model.", args.verbose_model_loading)

    log_model_loading(f"Resolving JAX device: {args.device!r}", args.verbose_model_loading)
    device = resolve_jax_device(args.device, allow_cpu=args.allow_cpu)
    log_model_loading(f"Resolved JAX device: {device if device is not None else 'AlphaGenome default'}", args.verbose_model_loading)

    log_model_loading("Building organism/reference settings ...", args.verbose_model_loading)
    organism_settings = build_organism_settings(local_dna_model, args)
    if organism_settings is None:
        log_model_loading("Using AlphaGenome default reference settings.", args.verbose_model_loading)
    else:
        log_model_loading("Using user-provided local reference settings.", args.verbose_model_loading)

    if args.weights_source == "checkpoint":
        if not args.checkpoint_path:
            raise ValueError("--checkpoint-path is required with --weights-source checkpoint.")
        log_model_loading(f"Loading local AlphaGenome checkpoint: {args.checkpoint_path}", args.verbose_model_loading)
        model = call_with_supported_kwargs(
            local_dna_model.create,
            checkpoint_path=str(Path(args.checkpoint_path)),
            organism_settings=organism_settings,
            device=device,
        )
        elapsed = time.monotonic() - load_started
        log_model_loading(f"Finished local AlphaGenome checkpoint load in {elapsed:.1f}s.", args.verbose_model_loading)
        return model

    factory_name = {
        "huggingface": "create_from_huggingface",
        "kaggle": "create_from_kaggle",
    }[args.weights_source]
    if not hasattr(local_dna_model, factory_name):
        raise AttributeError(f"alphagenome_research.model.dna_model has no {factory_name}().")

    log_model_loading(
        f"Loading AlphaGenome {args.model_version!r} from {args.weights_source} via {factory_name}() ...",
        args.verbose_model_loading,
    )
    model = call_with_supported_kwargs(
        getattr(local_dna_model, factory_name),
        args.model_version,
        organism_settings=organism_settings,
        device=device,
    )
    elapsed = time.monotonic() - load_started
    log_model_loading(f"Finished AlphaGenome model load in {elapsed:.1f}s.", args.verbose_model_loading)
    return model


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
            "cancer types and score them with a local AlphaGenome Research model."
        )
    )
    parser.add_argument(
        "--weights-source",
        choices=["checkpoint", "huggingface", "kaggle"],
        default=DEFAULT_WEIGHTS_SOURCE,
        help="Where to load local AlphaGenome weights from (default: %(default)s).",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=DEFAULT_CHECKPOINT_PATH,
        help="Local AlphaGenome checkpoint directory; required only with --weights-source checkpoint.",
    )
    parser.add_argument(
        "--model-version",
        default=DEFAULT_MODEL_VERSION,
        help="Model version for --weights-source huggingface/kaggle (default: %(default)s).",
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help="JAX device for local inference: auto, gpu, tpu, cpu, or exact device string.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        default=DEFAULT_ALLOW_CPU,
        help="Allow local AlphaGenome to run on CPU. This is usually very slow.",
    )
    parser.add_argument(
        "--fasta-path",
        default=DEFAULT_FASTA_PATH,
        help="Optional local human FASTA path for fully offline interval extraction.",
    )
    parser.add_argument(
        "--gtf-feather-path",
        default=DEFAULT_GTF_FEATHER_PATH,
        help="Optional local GENCODE GTF feather path.",
    )
    parser.add_argument(
        "--pas-feather-path",
        default=DEFAULT_PAS_FEATHER_PATH,
        help="Optional local polyadenylation annotation feather path.",
    )
    parser.add_argument(
        "--splice-site-starts-feather-path",
        default=DEFAULT_SPLICE_SITE_STARTS_FEATHER_PATH,
        help="Optional local splice-site starts feather path.",
    )
    parser.add_argument(
        "--splice-site-ends-feather-path",
        default=DEFAULT_SPLICE_SITE_ENDS_FEATHER_PATH,
        help="Optional local splice-site ends feather path.",
    )
    parser.add_argument(
        "--calibration-path",
        default=DEFAULT_CALIBRATION_PATH,
        help="Optional local calibration scorer path if supported by alphagenome_research.",
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
        "--driver-gene-file",
        default=DEFAULT_DRIVER_GENE_FILE,
        help=argparse.SUPPRESS,
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
        type=str,
        default=DEFAULT_MAX_WORKERS,
        help="Max parallel local scoring workers, or 'auto' for visible GPU device count (default: %(default)s).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_false",
        dest="show_progress",
        default=DEFAULT_SHOW_PROGRESS,
        help="Disable per-batch scoring progress bars.",
    )
    parser.add_argument(
        "--quiet-model-loading",
        action="store_false",
        dest="verbose_model_loading",
        default=DEFAULT_VERBOSE_MODEL_LOADING,
        help="Disable timestamped local model loading status messages.",
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
    try:
        args.max_workers = resolve_max_workers(args.max_workers)
    except ValueError as exc:
        parser.error(str(exc))
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
    # Phase 2: Score each organ with local AlphaGenome
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Phase 2: Local AlphaGenome scoring")
    print("=" * 60)

    client = create_local_model(args)

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
        n_unique_rows = sum(1 for _ in open(unique_path)) - 1
        print(f"  {n_unique_rows:,} unique mutations to score")

        # Organ-wide tissue-specific track filter (union of all cancer types)
        tissue_curies = get_organ_ontology_curies(organ_map[organ], organ)

        completed_mutations = load_completed_mutations(completed_path, scored_path)
        if completed_mutations:
            print(f"  Resume state: {len(completed_mutations):,} completed mutation IDs loaded")

        # Parse and score in batches. This avoids materializing millions of
        # Variant/Interval objects before the first scoring call.
        total_rows_written = 0
        total_mutations_completed = 0
        total_parse_skipped = 0
        batch_size = args.batch_size
        n_batches = (n_unique_rows + batch_size - 1) // batch_size
        chunks = pd.read_csv(unique_path, sep="\t", dtype=str, chunksize=batch_size)
        for batch_idx, unique_batch in enumerate(chunks):
            lo = batch_idx * batch_size
            hi = min(lo + len(unique_batch), n_unique_rows)

            if completed_mutations:
                before_resume_filter = len(unique_batch)
                unique_batch = unique_batch[
                    ~unique_batch["mutation"].isin(completed_mutations)
                ]
                n_resume_skipped = before_resume_filter - len(unique_batch)
            else:
                n_resume_skipped = 0

            variants, intervals, mutation_meta = [], [], []
            n_batch_parse_skipped = 0
            for _, row in unique_batch.iterrows():
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
                    n_batch_parse_skipped += 1

            total_parse_skipped += n_batch_parse_skipped
            print(
                f"  Batch {batch_idx + 1}/{n_batches}: "
                f"rows {lo + 1}-{hi} of {n_unique_rows} "
                f"({len(variants):,} to score"
                f"{', ' + str(n_resume_skipped) + ' already completed' if n_resume_skipped else ''}"
                f"{', ' + str(n_batch_parse_skipped) + ' parse skipped' if n_batch_parse_skipped else ''})",
                flush=True,
            )
            if not variants:
                continue

            batch_rows, batch_completed = score_variants_batch(
                client=client,
                batch_variants=variants,
                batch_intervals=intervals,
                scorer_list=scorer_list,
                tissue_curies=tissue_curies,
                mutation_meta=mutation_meta,
                max_workers=args.max_workers,
                show_progress=args.show_progress,
                progress_label=f"{organ} batch {batch_idx + 1}/{n_batches}",
            )
            rows_written = append_score_rows(scored_path, batch_rows)
            append_completed_mutations(completed_path, batch_completed)
            completed_mutations.update(batch_completed)
            total_rows_written += rows_written
            total_mutations_completed += len(batch_completed)
            print(
                f"    → {rows_written:,} rows saved, "
                f"{len(batch_completed):,} mutations completed "
                f"(session total rows: {total_rows_written:,})"
            )

        if total_parse_skipped:
            print(f"  Skipped {total_parse_skipped:,} unparseable mutations.")
        if total_mutations_completed == 0:
            print(f"  No new valid variants completed for {organ}.")
        print(f"  Saved/resumed scores → {scored_path}")
        print(f"  Completion ledger → {completed_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
