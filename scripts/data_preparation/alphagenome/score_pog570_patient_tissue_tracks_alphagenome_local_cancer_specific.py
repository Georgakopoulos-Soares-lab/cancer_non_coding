#!/usr/bin/env python
"""Score POG570 patient mutations with local AlphaGenome tissue tracks and cancer-specific genes.

This standalone script does not import helper functions from sibling scripts.
It does the following:

  1. Takes one or more POG570 patient IDs from the command line.
  2. Uses matching lifted hg38 mutation files from data/PCAWG_POG570_mutations_hg38.
  3. Scores cancer-specific driver genes selected from
     metadata/pog570_patient_driver_gene_file_mapping.tsv.
  4. Looks up the patient's analysis cohort from
     metadata/pog570_patient_driver_gene_file_mapping.tsv.
  5. Selects tissue-specific AlphaGenome tracks using
     metadata/pog570_tissue_mapping.tsv.
  6. Writes compact AlphaGenome-style score rows and compressed WT/mutant
     tissue-specific prediction outputs for the selected tracks.

The score CSVs are the main downstream input. The compressed output NPZ files
contain the selected WT and mutant AlphaGenome track values used to calculate
those scores. 
"""

import argparse
import importlib
import inspect
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import pandas as pd

from alphagenome.data import genome
from alphagenome.models import dna_client, dna_output, variant_scorers
from alphagenome.models.dna_model import Organism

from tqdm.auto import tqdm


PROJECT_DIR = Path(__file__).resolve().parents[3]
POG570_MUTATION_DIR = PROJECT_DIR / "data" / "PCAWG_POG570_mutations_hg38"
DRIVER_GENE_DIR = PROJECT_DIR / "data" / "driver_genes_coords"
INTOGEN_DRIVER_GENE_DIR = PROJECT_DIR / "data" / "driver_genes_intogen_curated"
PANCANCER_DRIVER_GENE_FILE = DRIVER_GENE_DIR / "Pancancer_1pc.tsv"
DEFAULT_DRIVER_GENE_FILE = PANCANCER_DRIVER_GENE_FILE
TISSUE_TRACK_MAPPING = PROJECT_DIR / "metadata" / "tissue_alphagenome_tracks.tsv"
POG570_TISSUE_MAPPING = PROJECT_DIR / "metadata" / "pog570_tissue_mapping.tsv"
POG570_PATIENT_COHORT_MAPPING = PROJECT_DIR / "metadata" / "pog570_patient_analysis_cohort_mapping.tsv"
POG570_PATIENT_DRIVER_GENE_MAPPING = PROJECT_DIR / "metadata" / "pog570_patient_driver_gene_file_mapping.tsv"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data" / "alphagenome_scores" / "pog570_patient_tissue_tracks_local_cancer_specific"
ALPHAGENOME_REFERENCE_DIR = PROJECT_DIR / "data" / "alphagenome_reference" / "hg38"
ALPHAGENOME_RESEARCH_SRC = PROJECT_DIR / "alphagenome_research" / "src"

DEFAULT_WEIGHTS_SOURCE = "huggingface"
DEFAULT_CHECKPOINT_PATH = None
DEFAULT_MODEL_VERSION = "all_folds"
DEFAULT_DEVICE = "gpu"
DEFAULT_ALLOW_CPU = False
DEFAULT_FASTA_PATH = str(ALPHAGENOME_REFERENCE_DIR / "GRCh38.p13.genome.fa")
DEFAULT_GTF_FEATHER_PATH = str(ALPHAGENOME_REFERENCE_DIR / "gencode.v46.annotation.gtf.gz.feather")
DEFAULT_PAS_FEATHER_PATH = str(ALPHAGENOME_REFERENCE_DIR / "polyadb_human_v3_exon3_contiguous_gtfv46.feather")
DEFAULT_SPLICE_SITE_STARTS_FEATHER_PATH = str(ALPHAGENOME_REFERENCE_DIR / "gencode.v46.splice_sites_starts.feather")
DEFAULT_SPLICE_SITE_ENDS_FEATHER_PATH = str(ALPHAGENOME_REFERENCE_DIR / "gencode.v46.splice_sites_ends.feather")
DEFAULT_CALIBRATION_PATH = str(ALPHAGENOME_REFERENCE_DIR / "calibration_scores.pb")
DEFAULT_SEQUENCE_LENGTH_NAME = "1mb"
DEFAULT_AGGREGATION_TYPES = ["DIFF_MEAN", "L2_DIFF"]
DEFAULT_EXCLUDED_OUTPUT_TYPES = ["SPLICE_JUNCTIONS"]
DEFAULT_SCORE_REGION = "gene"
DEFAULT_BATCH_SIZE = 1
DEFAULT_MAX_CANCER_DRIVER_GENES = 30
DEFAULT_MAX_GENES = None
DEFAULT_MAX_TRACKS_PER_OUTPUT = 4
DEFAULT_FORCE = False
DEFAULT_SHOW_PROGRESS = True
DEFAULT_VERBOSE_MODEL_LOADING = True
DEFAULT_FASTA_URL = "https://storage.googleapis.com/alphagenome/reference/gencode/hg38/GRCh38.p13.genome.fa"
GENES = None
INCLUDE_UNMUTATED_GENES = False
SAVE_TISSUE_OUTPUTS = True

INTOGEN_DRIVER_GENE_ALIASES = {
    "KIRP": ["KIRP_KIRC", "KIRP_KICH"],
    "SARC": ["SARC_LMS", "SARC_LIPO", "SARC_OS", "SARC_ES", "SARC_ANGS"],
    "SKCM": ["SKCM_ALL"],
    "TGCT": ["TGCT_MGCT"],
}

SEQUENCE_LENGTHS = {
    "16kb": dna_client.SEQUENCE_LENGTH_16KB,
    "100kb": dna_client.SEQUENCE_LENGTH_100KB,
    "500kb": dna_client.SEQUENCE_LENGTH_500KB,
    "1mb": dna_client.SEQUENCE_LENGTH_1MB,
}

COMPLETED_SUFFIX = ".completed_patient_genes.tsv"
PATIENT_OUTPUT_DIR_NAME = "patients"
RAW_OUTPUT_DIR_NAME = "tissue_outputs"
PREFERRED_COLUMNS = [
    "patient_id", "cohort", "cancer_type", "gene", "gene_strand", "output_type",
    "scorer", "aggregation_type", "score_region", "score", "n_tracks",
    "n_values", "n_aligned_bins", "n_window_mutations",
    "track_name", "track_strand", "ontology_curie", "biosample_name",
    "assay_title", "selected_strand", "mutations", "hg19_mutations_lifted_to_hg38",
    "interval_chrom", "interval_start", "interval_end", "score_start", "score_end",
    "mean_abs_delta", "max_abs_delta", "mean_delta",
]

def log_model_loading(message: str, enabled: bool = True) -> None:
    if enabled:
        print(f"[model load {time.strftime('%H:%M:%S')}] {message}", flush=True)


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
        raise ValueError("Refusing to run AlphaGenome on CPU; set DEFAULT_ALLOW_CPU = True to opt in.")
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


def build_organism_settings(local_dna_model):
    """Build optional local reference-file settings for Homo sapiens."""
    reference_kwargs = {
        "fasta_path": DEFAULT_FASTA_PATH,
        "gtf_feather_path": DEFAULT_GTF_FEATHER_PATH,
        "pas_feather_path": DEFAULT_PAS_FEATHER_PATH,
        "splice_site_starts_feather_path": DEFAULT_SPLICE_SITE_STARTS_FEATHER_PATH,
        "splice_site_ends_feather_path": DEFAULT_SPLICE_SITE_ENDS_FEATHER_PATH,
        "calibration_path": DEFAULT_CALIBRATION_PATH,
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
            "update alphagenome_research or use the huggingface/kaggle weights source."
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


def create_local_model():
    """Create an AlphaGenome Research model for local scoring."""
    load_started = time.monotonic()
    log_model_loading("Starting local AlphaGenome model setup.", DEFAULT_VERBOSE_MODEL_LOADING)

    if ALPHAGENOME_RESEARCH_SRC.exists() and str(ALPHAGENOME_RESEARCH_SRC) not in sys.path:
        sys.path.insert(0, str(ALPHAGENOME_RESEARCH_SRC))
        log_model_loading(
            f"Added AlphaGenome Research source path: {ALPHAGENOME_RESEARCH_SRC}",
            DEFAULT_VERBOSE_MODEL_LOADING,
        )

    log_model_loading("Importing alphagenome_research.model.dna_model ...", DEFAULT_VERBOSE_MODEL_LOADING)
    local_dna_model = importlib.import_module("alphagenome_research.model.dna_model")
    log_model_loading("Imported alphagenome_research.model.dna_model.", DEFAULT_VERBOSE_MODEL_LOADING)

    log_model_loading(f"Resolving JAX device: {DEFAULT_DEVICE!r}", DEFAULT_VERBOSE_MODEL_LOADING)
    device = resolve_jax_device(DEFAULT_DEVICE, allow_cpu=DEFAULT_ALLOW_CPU)
    log_model_loading(f"Resolved JAX device: {device if device is not None else 'AlphaGenome default'}", DEFAULT_VERBOSE_MODEL_LOADING)

    log_model_loading("Building organism/reference settings ...", DEFAULT_VERBOSE_MODEL_LOADING)
    organism_settings = build_organism_settings(local_dna_model)
    if organism_settings is None:
        log_model_loading("Using AlphaGenome default reference settings.", DEFAULT_VERBOSE_MODEL_LOADING)
    else:
        log_model_loading("Using user-provided local reference settings.", DEFAULT_VERBOSE_MODEL_LOADING)

    if DEFAULT_WEIGHTS_SOURCE == "checkpoint":
        if not DEFAULT_CHECKPOINT_PATH:
            raise ValueError("DEFAULT_CHECKPOINT_PATH is required with checkpoint weights.")
        log_model_loading(f"Loading local AlphaGenome checkpoint: {DEFAULT_CHECKPOINT_PATH}", DEFAULT_VERBOSE_MODEL_LOADING)
        model = call_with_supported_kwargs(
            local_dna_model.create,
            checkpoint_path=str(Path(DEFAULT_CHECKPOINT_PATH)),
            organism_settings=organism_settings,
            device=device,
        )
        elapsed = time.monotonic() - load_started
        log_model_loading(f"Finished local AlphaGenome checkpoint load in {elapsed:.1f}s.", DEFAULT_VERBOSE_MODEL_LOADING)
        return model

    factory_name = {
        "huggingface": "create_from_huggingface",
        "kaggle": "create_from_kaggle",
    }[DEFAULT_WEIGHTS_SOURCE]
    if not hasattr(local_dna_model, factory_name):
        raise AttributeError(f"alphagenome_research.model.dna_model has no {factory_name}().")

    log_model_loading(
        f"Loading AlphaGenome {DEFAULT_MODEL_VERSION!r} from {DEFAULT_WEIGHTS_SOURCE} via {factory_name}() ...",
        DEFAULT_VERBOSE_MODEL_LOADING,
    )
    model = call_with_supported_kwargs(
        getattr(local_dna_model, factory_name),
        DEFAULT_MODEL_VERSION,
        organism_settings=organism_settings,
        device=device,
    )
    elapsed = time.monotonic() - load_started
    log_model_loading(f"Finished AlphaGenome model load in {elapsed:.1f}s.", DEFAULT_VERBOSE_MODEL_LOADING)
    return model


def is_remote_path(path: str | None) -> bool:
    return isinstance(path, str) and path.startswith(("http://", "https://", "gs://"))


def validate_reference_files() -> None:
    """Fail early when the default local AlphaGenome references are missing."""
    required = {
        "DEFAULT_FASTA_PATH": DEFAULT_FASTA_PATH,
        "DEFAULT_GTF_FEATHER_PATH": DEFAULT_GTF_FEATHER_PATH,
        "DEFAULT_PAS_FEATHER_PATH": DEFAULT_PAS_FEATHER_PATH,
        "DEFAULT_SPLICE_SITE_STARTS_FEATHER_PATH": DEFAULT_SPLICE_SITE_STARTS_FEATHER_PATH,
        "DEFAULT_SPLICE_SITE_ENDS_FEATHER_PATH": DEFAULT_SPLICE_SITE_ENDS_FEATHER_PATH,
        "DEFAULT_CALIBRATION_PATH": DEFAULT_CALIBRATION_PATH,
    }
    missing = [
        f"{flag} {path}"
        for flag, path in required.items()
        if path and not is_remote_path(path) and not Path(path).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing local AlphaGenome reference file(s):\n  "
            + "\n  ".join(missing)
            + "\nDownload them into data/alphagenome_reference/hg38 or update the constants at the top of this script."
        )


def get_fasta_extractor(client):
    """Return a FASTA extractor from the model, configured FASTA, or AlphaGenome default URL."""
    try:
        return client._get_fasta_extractor(Organism.HOMO_SAPIENS)
    except Exception as model_exc:
        fasta_path = DEFAULT_FASTA_PATH or DEFAULT_FASTA_URL
        local_fasta = importlib.import_module("alphagenome_research.io.fasta")
        try:
            extractor = local_fasta.FastaExtractor(fasta_path)
        except Exception as fasta_exc:
            raise RuntimeError(
                "Could not create a FASTA extractor for patient-level WT sequence "
                f"extraction from {fasta_path!r}."
            ) from fasta_exc
        print(
            "Warning: local model had no FASTA extractor; using "
            f"{fasta_path!r} for WT sequence extraction.",
            file=sys.stderr,
        )
        return extractor


def normalise_mutation(mutation: str) -> str:
    parts = str(mutation).split(":")
    if len(parts) != 4:
        return str(mutation)
    chrom, coord, ref, alt = parts
    if "-" not in coord:
        coord = f"{int(coord)}-{int(coord)}"
    return f"{chrom}:{coord}:{ref}:{alt}"


def normalise_chrom(chrom: str) -> str:
    chrom = str(chrom)
    return chrom[3:] if chrom.lower().startswith("chr") else chrom


def parse_mutation_interval(mutation: str):
    chrom, coords, ref, alt = normalise_mutation(mutation).split(":")
    start, end = map(int, coords.split("-"))
    start0 = min(start, end) - 1
    end0 = max(start, end)
    return normalise_chrom(chrom), start0, end0, ref.replace("-", ""), alt.replace("-", "")


def load_driver_genes(cancer_type: str, driver_gene_file: str | None = None) -> pd.DataFrame:
    path = Path(driver_gene_file) if driver_gene_file else DEFAULT_DRIVER_GENE_FILE
    if not path.exists() and driver_gene_file is None:
        path = DRIVER_GENE_DIR / f"{cancer_type}.tsv"
    if not path.exists():
        path = DRIVER_GENE_DIR / "Pancancer.tsv"
    if not path.exists():
        raise FileNotFoundError(f"No driver gene coordinate file found for {cancer_type}.")
    df = pd.read_csv(path, sep="\t", dtype=str)
    required = {"gene", "chr", "start", "end"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    df["start"] = pd.to_numeric(df["start"])
    df["end"] = pd.to_numeric(df["end"])
    if "strand" not in df.columns:
        df["strand"] = "."
    return df


def clean_output_type(value: str) -> str:
    value = str(value)
    if "." in value:
        value = value.rsplit(".", 1)[-1]
    return value.upper()


def normalized_track_strand(value: str) -> str:
    value = str(value).strip()
    return value if value in {"+", "-", "."} else "."


def available_strand_priority(desired: str) -> list[str]:
    desired = normalized_track_strand(desired)
    if desired in {"+", "-"}:
        return [desired, ".", "+" if desired == "-" else "-"]
    return [".", "+", "-"]


def choose_tracks_for_gene(
    cancer_tracks: pd.DataFrame,
    gene: str,
    gene_to_strand: dict[str, str],
    max_tracks_per_output: int,
) -> pd.DataFrame:
    desired = gene_to_strand.get(gene, "")
    selected = []
    for output_key, output_tracks in cancer_tracks.groupby("output_key", sort=True):
        chosen = output_tracks.iloc[0:0].copy()
        for strand in available_strand_priority(desired):
            strand_tracks = output_tracks[output_tracks["normalized_strand"] == strand].copy()
            if len(strand_tracks) > 0:
                chosen = strand_tracks
                break
        if len(chosen) > 0:
            selected.append(
                chosen.drop_duplicates("track_name")
                .head(max_tracks_per_output)
                .assign(selected_strand=chosen["normalized_strand"].iloc[0])
            )
    if not selected:
        return cancer_tracks.iloc[0:0].copy()
    return pd.concat(selected, ignore_index=True)


def alphagenome_output_type(output_key: str):
    return dna_output.OutputType[output_key]


def gene_interval(driver_gene_data: pd.DataFrame, gene: str, sequence_length: int) -> genome.Interval:
    row = driver_gene_data.loc[driver_gene_data["gene"] == gene].iloc[0]
    chrom = str(row["chr"])
    chrom = chrom if chrom.startswith("chr") else f"chr{chrom}"
    start = int(row["start"]) - 1
    end = int(row["end"])
    center = (start + end) // 2
    interval_start = max(0, center - sequence_length // 2)
    return genome.Interval(chromosome=chrom, start=interval_start, end=interval_start + sequence_length)


def apply_mutations(
    sequence: str,
    interval: genome.Interval,
    mutations: list[str],
) -> tuple[str, int, np.ndarray, np.ndarray, np.ndarray]:
    """Apply patient mutations and return mutant sequence plus ref->alt offset map.

    The returned map has one entry per reference-sequence base. Values are the
    corresponding mutant-sequence offset, or -1 for deleted/unmapped reference
    bases. Inserted mutant bases intentionally have no reference coordinate,
    so their mutant offsets and reference-anchor offsets are returned
    separately for scoring.
    """
    seq_chrom = normalise_chrom(interval.chromosome)
    edits = []
    for mutation in mutations:
        chrom, start0, end0, ref, alt = parse_mutation_interval(mutation)
        if chrom == seq_chrom and not (end0 <= interval.start or start0 >= interval.end):
            edits.append((start0 - interval.start, end0 - interval.start, ref, alt))
    edits = sorted(edits)

    non_overlapping = []
    last_end = -1
    for edit in edits:
        if edit[0] >= last_end:
            non_overlapping.append(edit)
            last_end = edit[1]

    ref_to_alt = np.full(len(sequence), -1, dtype=np.int32)
    inserted_alt_offsets = []
    inserted_ref_anchors = []
    mutated_parts = []
    ref_cursor = 0
    alt_cursor = 0
    applied = 0
    for start, _, ref, alt in non_overlapping:
        start = max(0, min(start, len(sequence)))
        end = max(start, min(start + len(ref), len(sequence)))

        if start < ref_cursor:
            continue

        if ref and sequence[start:end].upper() != ref.upper():
            print(
                f"  Reference mismatch at {interval.chromosome}:{interval.start + start + 1}; "
                f"expected {ref}, found {sequence[start:end]}",
                file=sys.stderr,
            )
            continue

        unchanged = sequence[ref_cursor:start]
        mutated_parts.append(unchanged)
        if unchanged:
            unchanged_len = len(unchanged)
            ref_to_alt[ref_cursor:start] = np.arange(
                alt_cursor, alt_cursor + unchanged_len, dtype=np.int32
            )
            alt_cursor += unchanged_len

        mutated_parts.append(alt)
        aligned_len = min(len(ref), len(alt), end - start)
        if aligned_len > 0:
            ref_to_alt[start:start + aligned_len] = np.arange(
                alt_cursor, alt_cursor + aligned_len, dtype=np.int32
            )
        if len(alt) > aligned_len:
            inserted_start = alt_cursor + aligned_len
            inserted_end = alt_cursor + len(alt)
            inserted_alt_offsets.extend(range(inserted_start, inserted_end))
            anchor = start if len(ref) > 0 else max(start - 1, 0)
            inserted_ref_anchors.extend([anchor] * (inserted_end - inserted_start))
        alt_cursor += len(alt)
        ref_cursor = end
        applied += 1

    tail = sequence[ref_cursor:]
    mutated_parts.append(tail)
    if tail:
        tail_len = len(tail)
        ref_to_alt[ref_cursor:] = np.arange(
            alt_cursor, alt_cursor + tail_len, dtype=np.int32
        )
        alt_cursor += tail_len

    mutated = "".join(mutated_parts)
    if len(mutated) > len(sequence):
        mutated = mutated[:len(sequence)]
        ref_to_alt[ref_to_alt >= len(sequence)] = -1
        inserted = [
            (alt_offset, ref_anchor)
            for alt_offset, ref_anchor in zip(inserted_alt_offsets, inserted_ref_anchors, strict=False)
            if alt_offset < len(sequence)
        ]
        inserted_alt_offsets = [alt_offset for alt_offset, _ in inserted]
        inserted_ref_anchors = [ref_anchor for _, ref_anchor in inserted]
    elif len(mutated) < len(sequence):
        mutated = mutated + "N" * (len(sequence) - len(mutated))
    return (
        mutated,
        applied,
        ref_to_alt,
        np.asarray(inserted_alt_offsets, dtype=np.int32),
        np.asarray(inserted_ref_anchors, dtype=np.int32),
    )


def filter_track_data_to_curated(track_data, curated_tracks: pd.DataFrame):
    if track_data is None or len(curated_tracks) == 0:
        return None
    if not hasattr(track_data, "select_tracks_by_index"):
        return None
    metadata = track_data.metadata.copy()
    if "name" not in metadata.columns:
        return None
    names = set(curated_tracks["track_name"].dropna().astype(str))
    keep_indices = np.flatnonzero(metadata["name"].astype(str).isin(names).to_numpy())
    if len(keep_indices) == 0:
        return None
    return track_data.select_tracks_by_index(keep_indices)


def track_values(track_data) -> np.ndarray:
    values = np.asarray(track_data.values)
    if values.ndim == 1:
        values = values[:, None]
    return values


def select_score_interval(interval: genome.Interval, gene_row: pd.Series, score_region: str) -> genome.Interval:
    if score_region == "full_interval":
        return interval
    gene_start = int(gene_row["start"]) - 1
    gene_end = int(gene_row["end"])
    start = max(interval.start, gene_start)
    end = min(interval.end, gene_end)
    if start >= end:
        return interval
    return genome.Interval(chromosome=interval.chromosome, start=start, end=end)


def align_track_values_by_ref_map(
    wt_track_data,
    mut_track_data,
    interval: genome.Interval,
    score_interval: genome.Interval,
    ref_to_alt: np.ndarray,
    inserted_alt_offsets: np.ndarray,
    inserted_ref_anchors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Return WT/ALT values aligned by homologous positions plus indel effects.

    Homologous positions are compared by the ref->alt coordinate map.
    Deleted REF positions contribute as REF vs zero ALT. Inserted ALT positions
    anchored in the scored interval contribute as zero REF vs ALT. This captures
    indel effects without letting indels shift all downstream comparisons.
    """
    wt_values = track_values(wt_track_data)
    mut_values = track_values(mut_track_data)
    if wt_values.ndim != 2 or mut_values.ndim != 2:
        return None

    resolution = int(getattr(wt_track_data, "resolution", 1) or 1)
    start_offset = max(0, int(score_interval.start) - int(interval.start))
    end_offset = min(len(ref_to_alt), int(score_interval.end) - int(interval.start))
    if start_offset >= end_offset:
        return None

    ref_offsets = np.arange(start_offset, end_offset, dtype=np.int64)
    alt_offsets = ref_to_alt[ref_offsets].astype(np.int64)

    aligned_keep = alt_offsets >= 0
    aligned_ref_bins = ref_offsets[aligned_keep] // resolution
    aligned_alt_bins = alt_offsets[aligned_keep] // resolution
    aligned_keep_bins = (
        (aligned_ref_bins >= 0)
        & (aligned_ref_bins < wt_values.shape[0])
        & (aligned_alt_bins >= 0)
        & (aligned_alt_bins < mut_values.shape[0])
    )
    aligned_pairs = np.empty((0, 2), dtype=np.int64)
    if np.any(aligned_keep_bins):
        aligned_pairs = np.stack(
            [aligned_ref_bins[aligned_keep_bins], aligned_alt_bins[aligned_keep_bins]],
            axis=1,
        )
        aligned_pairs = np.unique(aligned_pairs, axis=0)

    deleted_ref_offsets = ref_offsets[alt_offsets < 0]
    deleted_ref_bins = deleted_ref_offsets // resolution
    deleted_ref_bins = np.unique(
        deleted_ref_bins[
            (deleted_ref_bins >= 0) & (deleted_ref_bins < wt_values.shape[0])
        ]
    )

    if len(inserted_alt_offsets) and len(inserted_ref_anchors):
        inserted_keep = (
            (inserted_ref_anchors >= start_offset)
            & (inserted_ref_anchors < end_offset)
        )
        inserted_alt_bins = inserted_alt_offsets[inserted_keep].astype(np.int64) // resolution
        inserted_alt_bins = np.unique(
            inserted_alt_bins[
                (inserted_alt_bins >= 0) & (inserted_alt_bins < mut_values.shape[0])
            ]
        )
    else:
        inserted_alt_bins = np.asarray([], dtype=np.int64)

    rows_ref = []
    rows_alt = []
    if len(aligned_pairs):
        rows_ref.append(wt_values[aligned_pairs[:, 0]])
        rows_alt.append(mut_values[aligned_pairs[:, 1]])
    if len(deleted_ref_bins):
        deleted_ref_values = wt_values[deleted_ref_bins]
        rows_ref.append(deleted_ref_values)
        rows_alt.append(np.zeros_like(deleted_ref_values))
    if len(inserted_alt_bins):
        inserted_alt_values = mut_values[inserted_alt_bins]
        rows_ref.append(np.zeros_like(inserted_alt_values))
        rows_alt.append(inserted_alt_values)

    if not rows_ref:
        return None

    aligned_ref = np.concatenate(rows_ref, axis=0)
    aligned_alt = np.concatenate(rows_alt, axis=0)
    return aligned_ref, aligned_alt, int(aligned_ref.shape[0])


def parse_aggregation_types(names: list[str]) -> list[variant_scorers.AggregationType]:
    aggregations = []
    missing = []
    for name in names:
        key = str(name).upper()
        try:
            aggregations.append(variant_scorers.AggregationType[key])
        except KeyError:
            missing.append(name)
    if missing:
        available = ", ".join(item.name for item in variant_scorers.AggregationType)
        raise ValueError(f"Unknown aggregation type(s): {missing}. Available: {available}")
    return aggregations


def aggregate_ref_alt_scores(
    ref_values: np.ndarray,
    alt_values: np.ndarray,
    aggregation_type: variant_scorers.AggregationType,
) -> np.ndarray:
    positional_axes = tuple(range(ref_values.ndim - 1))
    if aggregation_type == variant_scorers.AggregationType.DIFF_MEAN:
        return np.nanmean(alt_values, axis=positional_axes) - np.nanmean(ref_values, axis=positional_axes)
    if aggregation_type == variant_scorers.AggregationType.ACTIVE_MEAN:
        return np.maximum(
            np.nanmean(alt_values, axis=positional_axes),
            np.nanmean(ref_values, axis=positional_axes),
        )
    if aggregation_type == variant_scorers.AggregationType.DIFF_SUM:
        return np.nansum(alt_values, axis=positional_axes) - np.nansum(ref_values, axis=positional_axes)
    if aggregation_type == variant_scorers.AggregationType.ACTIVE_SUM:
        return np.maximum(
            np.nansum(alt_values, axis=positional_axes),
            np.nansum(ref_values, axis=positional_axes),
        )
    if aggregation_type == variant_scorers.AggregationType.L2_DIFF:
        return np.sqrt(np.nansum((alt_values - ref_values) ** 2, axis=positional_axes))
    if aggregation_type == variant_scorers.AggregationType.L2_DIFF_LOG1P:
        return np.sqrt(
            np.nansum((np.log1p(alt_values) - np.log1p(ref_values)) ** 2, axis=positional_axes)
        )
    if aggregation_type == variant_scorers.AggregationType.DIFF_SUM_LOG2:
        return np.nansum(np.log2(alt_values + 1), axis=positional_axes) - np.nansum(
            np.log2(ref_values + 1), axis=positional_axes
        )
    if aggregation_type == variant_scorers.AggregationType.DIFF_LOG2_SUM:
        return np.log2(1 + np.nansum(alt_values, axis=positional_axes)) - np.log2(
            1 + np.nansum(ref_values, axis=positional_axes)
        )
    raise ValueError(f"Unsupported aggregation type: {aggregation_type}")


def summarize_delta(
    patient_id: str,
    cancer_type: str,
    gene: str,
    gene_strand: str,
    interval: genome.Interval,
    score_interval: genome.Interval,
    mutations: list[str],
    ref_to_alt: np.ndarray,
    inserted_alt_offsets: np.ndarray,
    inserted_ref_anchors: np.ndarray,
    wt_output,
    mut_output,
    tracks: pd.DataFrame,
    aggregation_types: list[variant_scorers.AggregationType],
    score_region: str,
) -> list[dict]:
    rows = []
    combined_scores_by_aggregation: dict[str, list[float]] = {
        aggregation.name: [] for aggregation in aggregation_types
    }
    for output_key, output_tracks in tracks.groupby("output_key", sort=True):
        output_type = alphagenome_output_type(output_key)
        wt_track_data = filter_track_data_to_curated(wt_output.get(output_type), output_tracks)
        mut_track_data = filter_track_data_to_curated(mut_output.get(output_type), output_tracks)
        if wt_track_data is None or mut_track_data is None:
            continue

        aligned = align_track_values_by_ref_map(
            wt_track_data=wt_track_data,
            mut_track_data=mut_track_data,
            interval=interval,
            score_interval=score_interval,
            ref_to_alt=ref_to_alt,
            inserted_alt_offsets=inserted_alt_offsets,
            inserted_ref_anchors=inserted_ref_anchors,
        )
        if aligned is None:
            continue
        wt_values, mut_values, n_aligned_bins = aligned
        delta = mut_values.astype(np.float32) - wt_values.astype(np.float32)
        abs_delta = np.abs(delta)
        mean_abs_delta = float(np.nanmean(abs_delta))
        max_abs_delta = float(np.nanmax(abs_delta))
        mean_delta = float(np.nanmean(delta))

        for aggregation_type in aggregation_types:
            track_scores = aggregate_ref_alt_scores(wt_values, mut_values, aggregation_type)
            output_score = float(np.nanmean(np.abs(track_scores)))
            combined_scores_by_aggregation[aggregation_type.name].append(output_score)

            for track_idx, (_, track_meta) in enumerate(wt_track_data.metadata.reset_index(drop=True).iterrows()):
                track_score = float(track_scores[track_idx])
                track_delta = delta[..., track_idx]
                track_abs_delta = np.abs(track_delta)
                rows.append({
                    "patient_id": patient_id,
                    "cancer_type": cancer_type,
                    "gene": gene,
                    "gene_strand": gene_strand,
                    "output_type": output_key,
                    "scorer": "PATIENT_SINGLE_TRACK_SCORER",
                    "aggregation_type": aggregation_type.name,
                    "score_region": score_region,
                    "score": track_score,
                    "n_tracks": 1,
                    "n_values": int(track_delta.size),
                    "n_aligned_bins": n_aligned_bins,
                    "n_window_mutations": len(mutations),
                    "track_name": str(track_meta.get("name", "")),
                    "track_strand": str(track_meta.get("strand", "")),
                    "ontology_curie": str(track_meta.get("ontology_curie", "")),
                    "biosample_name": str(track_meta.get("biosample_name", "")),
                    "assay_title": str(track_meta.get("assay_title", track_meta.get("Assay title", ""))),
                    "selected_strand": str(output_tracks.get("selected_strand", pd.Series([""])).iloc[0]),
                    "mutations": ";".join(mutations),
                    "interval_chrom": interval.chromosome,
                    "interval_start": int(interval.start),
                    "interval_end": int(interval.end),
                    "score_start": int(score_interval.start),
                    "score_end": int(score_interval.end),
                    "mean_abs_delta": float(np.nanmean(track_abs_delta)),
                    "max_abs_delta": float(np.nanmax(track_abs_delta)),
                    "mean_delta": float(np.nanmean(track_delta)),
                })

            rows.append({
                "patient_id": patient_id,
                "cancer_type": cancer_type,
                "gene": gene,
                "gene_strand": gene_strand,
                "output_type": output_key,
                "scorer": "PATIENT_OUTPUT_TRACK_SCORER",
                "aggregation_type": aggregation_type.name,
                "score_region": score_region,
                "score": output_score,
                "n_tracks": int(delta.shape[-1]) if delta.ndim else 1,
                "n_values": int(delta.size),
                "n_aligned_bins": n_aligned_bins,
                "n_window_mutations": len(mutations),
                "track_name": "",
                "track_strand": "",
                "ontology_curie": "",
                "biosample_name": "",
                "assay_title": "",
                "selected_strand": str(output_tracks.get("selected_strand", pd.Series([""])).iloc[0]),
                "mutations": ";".join(mutations),
                "interval_chrom": interval.chromosome,
                "interval_start": int(interval.start),
                "interval_end": int(interval.end),
                "score_start": int(score_interval.start),
                "score_end": int(score_interval.end),
                "mean_abs_delta": mean_abs_delta,
                "max_abs_delta": max_abs_delta,
                "mean_delta": mean_delta,
            })

    for aggregation_name, output_scores in combined_scores_by_aggregation.items():
        if not output_scores:
            continue
        rows.append({
            "patient_id": patient_id,
            "cancer_type": cancer_type,
            "gene": gene,
            "gene_strand": gene_strand,
            "output_type": "COMBINED",
            "scorer": "PATIENT_COMBINED_TRACK_SCORER",
            "aggregation_type": aggregation_name,
            "score_region": score_region,
            "score": float(np.mean(output_scores)),
            "n_tracks": "",
            "n_values": "",
            "n_aligned_bins": "",
            "n_window_mutations": len(mutations),
            "mutations": ";".join(mutations),
            "interval_chrom": interval.chromosome,
            "interval_start": int(interval.start),
            "interval_end": int(interval.end),
            "score_start": int(score_interval.start),
            "score_end": int(score_interval.end),
            "mean_abs_delta": "",
            "max_abs_delta": "",
            "mean_delta": "",
        })
    return rows


def append_rows(path: Path, rows: list[dict]) -> int:
    if not rows:
        return 0
    df = pd.DataFrame(rows)
    cols = [c for c in PREFERRED_COLUMNS if c in df.columns]
    cols.extend(c for c in df.columns if c not in cols)
    df[cols].to_csv(path, index=False, mode="a", header=not path.exists() or path.stat().st_size == 0)
    return len(df)


def load_completed(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    df = pd.read_csv(path, sep="\t", dtype=str)
    if not {"patient_id", "gene"}.issubset(df.columns):
        return set()
    return set(zip(df["patient_id"], df["gene"], strict=False))


def append_completed(path: Path, patient_id: str, genes: list[str]) -> None:
    if not genes:
        return
    df = pd.DataFrame({"patient_id": patient_id, "gene": genes})
    df.to_csv(path, sep="\t", index=False, mode="a", header=not path.exists() or path.stat().st_size == 0)


def safe_patient_filename(patient_id: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in patient_id)


def progress(iterable, total: int, desc: str, enabled: bool):
    return tqdm(iterable, total=total, desc=desc, unit="gene")


def load_pog570_tissue_mapping(path: Path) -> pd.DataFrame:
    mapping = pd.read_csv(path, sep="\t", dtype=str)
    return mapping[["analysis_cohort", "alphagenome_tissue"]].drop_duplicates("analysis_cohort")


def tissue_for_cohort(analysis_cohort: str, tissue_mapping_path: Path) -> str:
    mapping = load_pog570_tissue_mapping(tissue_mapping_path)
    matches = mapping[mapping["analysis_cohort"].eq(analysis_cohort)]["alphagenome_tissue"].dropna()
    if matches.empty or not str(matches.iloc[0]).strip():
        raise ValueError(f"No AlphaGenome tissue mapping for analysis cohort {analysis_cohort!r}.")
    return str(matches.iloc[0])


def cohort_for_patient(patient_id: str, patient_mapping_path: Path) -> str:
    mapping = pd.read_csv(patient_mapping_path, sep="\t", dtype=str)
    matches = mapping[mapping["patient_id"].astype(str).eq(str(patient_id))]["analysis_cohort"].dropna()
    if matches.empty or not str(matches.iloc[0]).strip():
        raise ValueError(f"No analysis cohort mapping for patient {patient_id!r}.")
    return str(matches.iloc[0])


def driver_gene_files_for_patient(patient_id: str, mapping_path: Path) -> tuple[str, list[Path]]:
    mapping = pd.read_csv(mapping_path, sep="\t", dtype=str).fillna("")
    matches = mapping[mapping["patient_id"].astype(str).eq(str(patient_id))]
    if matches.empty:
        raise ValueError(f"No cancer-specific driver gene mapping for patient {patient_id!r}.")

    row = matches.iloc[0]
    driver_files = []
    for value in str(row["driver_gene_files"]).split(";"):
        if value.strip():
            path = Path(value)
            driver_files.append(path if path.is_absolute() else PROJECT_DIR / path)
    if not driver_files:
        raise ValueError(f"No cancer-specific driver gene files for patient {patient_id!r}.")
    return str(row["analysis_cohort"]), driver_files


def load_patient_driver_genes(driver_gene_files: list[Path]) -> pd.DataFrame:
    frames = []
    for path in driver_gene_files:
        if not path.exists():
            raise FileNotFoundError(f"Missing mapped driver gene file: {path}")
        df = load_driver_genes(path.stem, str(path))
        df["source_driver_cancer_type"] = path.stem
        frames.append(df)
    return pd.concat(frames, ignore_index=True).drop_duplicates("gene")


def intogen_files_for_driver_cancer_type(driver_cancer_type: str) -> list[Path]:
    names = [driver_cancer_type]
    names.extend(INTOGEN_DRIVER_GENE_ALIASES.get(driver_cancer_type, []))
    return [
        path
        for name in names
        for path in [INTOGEN_DRIVER_GENE_DIR / f"{name}.tsv"]
        if path.exists()
    ]


def load_intogen_gene_sample_percent(driver_cancer_types: list[str]) -> dict[str, float]:
    ranks = {}
    for cancer_type in driver_cancer_types:
        for path in intogen_files_for_driver_cancer_type(cancer_type):
            df = pd.read_csv(path, sep="\t", dtype=str)
            if not {"Symbol", "Samples (%)"}.issubset(df.columns):
                continue
            df["sample_percent"] = pd.to_numeric(
                df["Samples (%)"].str.replace(",", "", regex=False),
                errors="coerce",
            )
            for _, row in df.dropna(subset=["Symbol", "sample_percent"]).iterrows():
                gene = str(row["Symbol"])
                ranks[gene] = max(ranks.get(gene, float("-inf")), float(row["sample_percent"]))
    return ranks


def rank_multifile_driver_genes(driver_genes: pd.DataFrame, driver_gene_files: list[Path]) -> pd.DataFrame:
    if len(driver_gene_files) <= 1:
        return driver_genes

    ranks = load_intogen_gene_sample_percent([path.stem for path in driver_gene_files])
    if not ranks:
        return driver_genes

    ranked = driver_genes.copy()
    ranked["_intogen_sample_percent"] = ranked["gene"].map(ranks).fillna(float("-inf"))
    ranked["_original_order"] = range(len(ranked))
    ranked = ranked.sort_values(
        ["_intogen_sample_percent", "_original_order"],
        ascending=[False, True],
    )
    return ranked.drop(columns=["_intogen_sample_percent", "_original_order"])


def load_patient_variants(patient_id: str, mutation_dir: Path) -> pd.DataFrame:
    path = mutation_dir / f"{patient_id}.tsv"
    df = pd.read_csv(path, sep="\t", dtype=str)
    if "mutation" not in df.columns:
        raise ValueError(f"{path} must contain a mutation column")
    if "gene" in df.columns:
        df = df.rename(columns={"gene": "variant_gene"})

    df = df.dropna(subset=["mutation"]).copy()
    df["mutation"] = df["mutation"].map(normalise_mutation)
    if "hg19_mutation" not in df.columns:
        df["hg19_mutation"] = ""
    return df


def load_tissue_tracks(tissue: str, excluded_output_types: set[str]) -> pd.DataFrame:
    tracks = pd.read_csv(TISSUE_TRACK_MAPPING, sep="\t", dtype=str).fillna("")
    tracks = tracks[tracks["tissue"].eq(tissue)].copy()
    if tracks.empty:
        return tracks
    tracks["output_key"] = tracks["output_type"].map(clean_output_type)
    tracks = tracks[~tracks["output_key"].isin(excluded_output_types)].copy()
    tracks["normalized_strand"] = tracks["strand"].map(normalized_track_strand)
    return tracks


def patient_output_paths(output_dir: Path, patient_id: str) -> tuple[Path, Path]:
    patient_dir = output_dir / PATIENT_OUTPUT_DIR_NAME
    patient_dir.mkdir(parents=True, exist_ok=True)
    patient_name = safe_patient_filename(patient_id)
    return (
        patient_dir / f"{patient_name}.csv",
        patient_dir / f"{patient_name}{COMPLETED_SUFFIX}",
    )


def raw_output_path(output_dir: Path, patient_id: str, gene: str) -> Path:
    path = output_dir / RAW_OUTPUT_DIR_NAME / safe_patient_filename(patient_id)
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{safe_patient_filename(gene)}.npz"


def save_tissue_outputs(path: Path, wt_output, mut_output, tracks: pd.DataFrame) -> None:
    arrays = {}
    metadata = {}
    for output_key, output_tracks in tracks.groupby("output_key", sort=True):
        output_type = alphagenome_output_type(output_key)
        wt_track_data = filter_track_data_to_curated(wt_output.get(output_type), output_tracks)
        mut_track_data = filter_track_data_to_curated(mut_output.get(output_type), output_tracks)
        if wt_track_data is None or mut_track_data is None:
            continue
        arrays[f"{output_key}__wt"] = track_values(wt_track_data).astype(np.float32)
        arrays[f"{output_key}__mut"] = track_values(mut_track_data).astype(np.float32)
        metadata[f"{output_key}__track_names"] = wt_track_data.metadata["name"].astype(str).to_numpy()
        metadata[f"{output_key}__track_strands"] = wt_track_data.metadata.get("strand", pd.Series([""])).astype(str).to_numpy()
    if arrays:
        np.savez_compressed(path, **arrays, **metadata)


def score_patient(
    client,
    fasta_extractor,
    patient_id: str,
    analysis_cohort: str,
    tissue: str,
    driver_gene_files: list[Path],
    output_dir: Path,
) -> None:
    output_path, completed_path = patient_output_paths(output_dir, patient_id)
    if DEFAULT_FORCE:
        for path in (output_path, completed_path):
            if path.exists():
                path.unlink()

    patient_variants = load_patient_variants(patient_id, POG570_MUTATION_DIR)
    driver_genes = load_patient_driver_genes(driver_gene_files)
    if GENES:
        driver_genes = driver_genes[driver_genes["gene"].isin(GENES)].copy()
    driver_genes = rank_multifile_driver_genes(driver_genes, driver_gene_files)
    if DEFAULT_MAX_CANCER_DRIVER_GENES:
        driver_genes = driver_genes.head(DEFAULT_MAX_CANCER_DRIVER_GENES).copy()
    if DEFAULT_MAX_GENES:
        driver_genes = driver_genes.head(DEFAULT_MAX_GENES).copy()

    gene_to_strand = dict(zip(driver_genes["gene"], driver_genes["strand"], strict=False))
    excluded_output_types = {clean_output_type(value) for value in DEFAULT_EXCLUDED_OUTPUT_TYPES}
    tissue_tracks = load_tissue_tracks(tissue, excluded_output_types)
    if tissue_tracks.empty:
        print(f"[{patient_id}] No AlphaGenome tracks for tissue {tissue!r}, skipping.", file=sys.stderr)
        return

    completed = load_completed(completed_path)
    genes = driver_genes["gene"].dropna().astype(str).tolist()
    gene_rows = {str(row["gene"]): row for _, row in driver_genes.iterrows()}
    driver_labels = ",".join(path.stem for path in driver_gene_files)
    rank_note = ", IntOGen-ranked" if len(driver_gene_files) > 1 else ""
    print(
        f"[{patient_id}] {analysis_cohort}/{tissue}: "
        f"{len(genes)} genes from {driver_labels}{rank_note}, "
        f"{len(patient_variants)} hg38 variants"
    )

    rows_written = 0
    completed_now = []
    sequence_length = SEQUENCE_LENGTHS[DEFAULT_SEQUENCE_LENGTH_NAME]
    aggregation_types = parse_aggregation_types(DEFAULT_AGGREGATION_TYPES)
    for gene in progress(genes, total=len(genes), desc=f"{patient_id}", enabled=DEFAULT_SHOW_PROGRESS):
        if (patient_id, gene) in completed:
            continue

        tracks = choose_tracks_for_gene(tissue_tracks, gene, gene_to_strand, DEFAULT_MAX_TRACKS_PER_OUTPUT)
        if tracks.empty:
            completed_now.append(gene)
            continue

        interval = gene_interval(driver_genes, gene, sequence_length)
        score_interval = select_score_interval(interval, gene_rows[gene], DEFAULT_SCORE_REGION)
        mutations = []
        seq_chrom = normalise_chrom(interval.chromosome)
        for mutation in patient_variants["mutation"]:
            chrom, start0, end0, _, _ = parse_mutation_interval(mutation)
            if chrom == seq_chrom and not (end0 <= interval.start or start0 >= interval.end):
                mutations.append(mutation)
        mutations = sorted(set(mutations))
        if not mutations and not INCLUDE_UNMUTATED_GENES:
            completed_now.append(gene)
            continue

        output_types = [alphagenome_output_type(key) for key in sorted(tracks["output_key"].unique())]
        try:
            wt_seq = fasta_extractor.extract(interval)
            mut_seq, n_applied, ref_to_alt, inserted_alt_offsets, inserted_ref_anchors = apply_mutations(
                wt_seq,
                interval,
                mutations,
            )
            if mutations and n_applied == 0:
                print(f"  [{patient_id} {gene}] No mutations applied after reference checks.", file=sys.stderr)

            wt_output = client.predict_sequence(
                wt_seq,
                organism=Organism.HOMO_SAPIENS,
                requested_outputs=output_types,
                ontology_terms=None,
                interval=interval,
            )
            mut_output = client.predict_sequence(
                mut_seq,
                organism=Organism.HOMO_SAPIENS,
                requested_outputs=output_types,
                ontology_terms=None,
                interval=interval,
            )
            if SAVE_TISSUE_OUTPUTS:
                save_tissue_outputs(raw_output_path(output_dir, patient_id, gene), wt_output, mut_output, tracks)
            rows = summarize_delta(
                patient_id=patient_id,
                cancer_type=analysis_cohort,
                gene=gene,
                gene_strand=gene_to_strand.get(gene, ""),
                interval=interval,
                score_interval=score_interval,
                mutations=mutations,
                ref_to_alt=ref_to_alt,
                inserted_alt_offsets=inserted_alt_offsets,
                inserted_ref_anchors=inserted_ref_anchors,
                wt_output=wt_output,
                mut_output=mut_output,
                tracks=tracks,
                aggregation_types=aggregation_types,
                score_region=DEFAULT_SCORE_REGION,
            )
            for row in rows:
                row["cohort"] = "POG570"
                row["hg19_mutations_lifted_to_hg38"] = ";".join(
                    patient_variants.loc[patient_variants["mutation"].isin(mutations), "hg19_mutation"].astype(str)
                )
        except Exception as exc:
            print(f"  Error scoring {patient_id} {gene}: {exc}", file=sys.stderr)
            continue

        rows_written += append_rows(output_path, rows)
        completed_now.append(gene)
        if len(completed_now) >= DEFAULT_BATCH_SIZE:
            append_completed(completed_path, patient_id, completed_now)
            completed_now = []

    append_completed(completed_path, patient_id, completed_now)
    print(f"[{patient_id}] wrote {rows_written:,} score rows -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score POG570 hg19 patient mutation files on cancer-specific driver genes."
    )
    parser.add_argument("patient_ids", nargs="+")
    cli_args = parser.parse_args()

    patient_ids = list(dict.fromkeys(str(patient_id) for patient_id in cli_args.patient_ids))
    validate_reference_files()

    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    patient_inputs = {}
    for patient_id in patient_ids:
        patient_path = POG570_MUTATION_DIR / f"{patient_id}.tsv"
        if not patient_path.is_file():
            raise FileNotFoundError(
                f"Missing lifted patient mutation file: {patient_path}. "
                "Run scripts/data_preparation/alphagenome/liftover_pog570_mutations.py first."
            )
        try:
            analysis_cohort, driver_gene_files = driver_gene_files_for_patient(
                patient_id,
                POG570_PATIENT_DRIVER_GENE_MAPPING,
            )
        except ValueError as exc:
            print(f"Skipping {patient_id}: {exc}", file=sys.stderr)
            continue
        try:
            tissue = tissue_for_cohort(analysis_cohort, POG570_TISSUE_MAPPING)
        except ValueError as exc:
            print(f"Skipping {patient_id}: {exc}", file=sys.stderr)
            continue
        patient_inputs[patient_id] = (analysis_cohort, tissue, driver_gene_files)

    patient_ids = [patient_id for patient_id in patient_ids if patient_id in patient_inputs]
    print(f"Patients to process: {len(patient_ids)}")
    print(f"Score outputs: {output_dir / PATIENT_OUTPUT_DIR_NAME}")
    print(f"Tissue outputs: {output_dir / RAW_OUTPUT_DIR_NAME}")
    if not patient_ids:
        print("No patients with usable tissue mappings.")
        return

    start = time.monotonic()
    client = create_local_model()
    fasta_extractor = get_fasta_extractor(client)

    for patient_id in patient_ids:
        analysis_cohort, tissue, driver_gene_files = patient_inputs[patient_id]
        print(f"Patient: {patient_id}; analysis cohort: {analysis_cohort}; AlphaGenome tissue: {tissue}")
        score_patient(
            client,
            fasta_extractor,
            patient_id,
            analysis_cohort,
            tissue,
            driver_gene_files,
            output_dir,
        )

    print(f"Done in {time.monotonic() - start:.1f}s.")


if __name__ == "__main__":
    main()
