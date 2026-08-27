#!/usr/bin/env python
"""Patient-level tissue-specific AlphaGenome scorer-style track scoring.

For each patient and driver gene from data/driver_genes_coords/Pancancer_1pc.tsv:

  1. Select cancer/tissue-specific AlphaGenome tracks from
     metadata/cancer_tissue_alphagenome_tracks.tsv.
  2. Build a gene-centered AlphaGenome interval.
  3. Apply all patient mutations that fall inside that interval to the
     reference sequence, producing one patient-mutated sequence per gene.
  4. Predict WT and patient-mutated tracks with the local AlphaGenome model.
  5. Apply AlphaGenome-style ref/alt aggregation logic over the gene span
     (or full interval) and save compact scorer rows, not raw tracks.

Unlike score_variant(), this script scores patient-level multi-mutation
sequences rather than one Variant at a time. Indels are handled during scoring
with a reference-to-mutant coordinate map:

  * homologous positions are compared as REF[i] vs ALT[mapped_i]
  * deleted reference positions contribute as REF[i] vs 0
  * inserted mutant positions contribute as 0 vs ALT[j]

This lets insertions/deletions contribute to the score without causing all
downstream positions to be compared at shifted indices.

Outputs:
  data/alphagenome_scores/patient_tissue_tracks_local/
    patients/{PATIENT_ID}.csv
    patients/{PATIENT_ID}.completed_patient_genes.tsv

Score rows:
  PATIENT_SINGLE_TRACK_SCORER     one row per patient/gene/output/track/scorer
  PATIENT_OUTPUT_TRACK_SCORER     one row per patient/gene/output/scorer
  PATIENT_COMBINED_TRACK_SCORER   one row per patient/gene/scorer

Examples:
  # Smoke test one patient and three genes on GPU.
  python scripts/data_preparation/alphagenome/score_patient_tissue_tracks_alphagenome_local.py \\
      --patients TCGA-02-0026 \\
      --genes TP53 PTEN EGFR \\
      --device gpu

  # Score a patient list, default aggregation types DIFF_MEAN and L2_DIFF.
  python scripts/data_preparation/alphagenome/score_patient_tissue_tracks_alphagenome_local.py \\
      --patient-list metadata/TCGA_patient_list.txt \\
      --device gpu \\
      --batch-size 5

  # Use specific AlphaGenome-style aggregation types over the gene span.
  python scripts/data_preparation/alphagenome/score_patient_tissue_tracks_alphagenome_local.py \\
      --patients TCGA-02-0026 \\
      --aggregation-types DIFF_MEAN DIFF_SUM L2_DIFF_LOG1P \\
      --score-region gene \\
      --device gpu

  # Score over the full AlphaGenome interval instead of just the gene span.
  python scripts/data_preparation/alphagenome/score_patient_tissue_tracks_alphagenome_local.py \\
      --patients TCGA-02-0026 \\
      --score-region full_interval \\
      --device gpu

  # Fully local/offline weights. Reference files are read from
  # data/alphagenome_reference/hg38 by default.
  python scripts/data_preparation/alphagenome/score_patient_tissue_tracks_alphagenome_local.py \\
      --patients TCGA-02-0026 \\
      --weights-source checkpoint \\
      --checkpoint-path /path/to/alphagenome-all-folds \\
      --device gpu
"""

import argparse
import hashlib
import json
import sys
import time
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

from alphagenome.data import genome, track_data
from alphagenome.models import dna_client, dna_output, variant_scorers
from alphagenome.models.dna_model import Organism

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from score_tissue_unique_variants_alphagenome_local import (
    DEFAULT_ALLOW_CPU,
    DEFAULT_CALIBRATION_PATH,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_DEVICE,
    DEFAULT_FASTA_PATH,
    DEFAULT_GTF_FEATHER_PATH,
    DEFAULT_MODEL_VERSION,
    DEFAULT_PAS_FEATHER_PATH,
    DEFAULT_SEQUENCE_LENGTH_NAME,
    DEFAULT_SPLICE_SITE_ENDS_FEATHER_PATH,
    DEFAULT_SPLICE_SITE_STARTS_FEATHER_PATH,
    DEFAULT_VERBOSE_MODEL_LOADING,
    DEFAULT_WEIGHTS_SOURCE,
    SEQUENCE_LENGTHS,
    create_local_model,
)


PROJECT_DIR = Path(__file__).resolve().parents[3]
PATIENT_CANCER_MAP = PROJECT_DIR / "data" / "tcga_patient_cancer_map.json"
PATIENT_VARIANTS_DIR = PROJECT_DIR / "data" / "TCGA" / "tcga_patient_variants"
DRIVER_GENE_DIR = PROJECT_DIR / "data" / "driver_genes_coords"
DEFAULT_DRIVER_GENE_FILE = DRIVER_GENE_DIR / "Pancancer_0.2pc.tsv"
TRACK_MAPPING = PROJECT_DIR / "metadata" / "cancer_tissue_alphagenome_tracks.tsv"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data" / "alphagenome_scores" / "patient_tissue_tracks_local"
ALPHAGENOME_REFERENCE_DIR = PROJECT_DIR / "data" / "alphagenome_reference" / "hg38"
DEFAULT_FASTA_PATH = str(ALPHAGENOME_REFERENCE_DIR / "GRCh38.p13.genome.fa")
DEFAULT_GTF_FEATHER_PATH = str(ALPHAGENOME_REFERENCE_DIR / "gencode.v46.annotation.gtf.gz.feather")
DEFAULT_PAS_FEATHER_PATH = str(ALPHAGENOME_REFERENCE_DIR / "polyadb_human_v3_exon3_contiguous_gtfv46.feather")
DEFAULT_SPLICE_SITE_STARTS_FEATHER_PATH = str(ALPHAGENOME_REFERENCE_DIR / "gencode.v46.splice_sites_starts.feather")
DEFAULT_SPLICE_SITE_ENDS_FEATHER_PATH = str(ALPHAGENOME_REFERENCE_DIR / "gencode.v46.splice_sites_ends.feather")
DEFAULT_CALIBRATION_PATH = str(ALPHAGENOME_REFERENCE_DIR / "calibration_scores.pb")
DEFAULT_FASTA_URL = (
    "https://storage.googleapis.com/alphagenome/reference/gencode/"
    "hg38/GRCh38.p13.genome.fa"
)

DEFAULT_PATIENTS = None
DEFAULT_PATIENT_LIST = None
DEFAULT_GENES = None
DEFAULT_MAX_GENES = None
DEFAULT_MAX_TRACKS_PER_OUTPUT = 4
DEFAULT_EXCLUDED_OUTPUT_TYPES = ["SPLICE_JUNCTIONS"]
DEFAULT_AGGREGATION_TYPES = ["DIFF_MEAN", "L2_DIFF"]
DEFAULT_SCORE_REGION = "gene"
DEFAULT_BATCH_SIZE = 4
DEFAULT_FORCE = False
DEFAULT_SHOW_PROGRESS = True

COMPLETED_SUFFIX = ".completed_patient_genes.tsv"
PATIENT_OUTPUT_DIR_NAME = "patients"
WT_CACHE_DIR_NAME = "wt_cache"
WT_CACHE_VERSION = "v2"
DRIVER_GENE_CACHE: dict[tuple[str, str], pd.DataFrame] = {}
CANCER_TRACK_CACHE: dict[tuple[str, tuple[str, ...]], pd.DataFrame] = {}
PREFERRED_COLUMNS = [
    "patient_id", "cancer_type", "gene", "gene_strand", "output_type",
    "scorer", "aggregation_type", "score_region", "score", "n_tracks",
    "n_values", "n_aligned_bins", "n_window_mutations",
    "track_name", "track_strand", "ontology_curie", "biosample_name",
    "assay_title", "selected_strand",
    "mutations", "interval_chrom", "interval_start", "interval_end",
    "score_start", "score_end",
    "mean_abs_delta", "max_abs_delta", "mean_delta",
]


def load_patient_cancer_map(path: Path) -> dict[str, str]:
    with path.open() as handle:
        raw = json.load(handle)
    return {str(patient): str(cancer).replace("TCGA-", "") for patient, cancer in raw.items()}


def is_remote_path(path: str | None) -> bool:
    return isinstance(path, str) and path.startswith(("http://", "https://", "gs://"))


def validate_reference_files(args, parser: argparse.ArgumentParser) -> None:
    """Fail early when the default local AlphaGenome references are missing."""
    required = {
        "--fasta-path": args.fasta_path,
        "--gtf-feather-path": args.gtf_feather_path,
        "--pas-feather-path": args.pas_feather_path,
        "--splice-site-starts-feather-path": args.splice_site_starts_feather_path,
        "--splice-site-ends-feather-path": args.splice_site_ends_feather_path,
        "--calibration-path": args.calibration_path,
    }
    missing = [
        f"{flag} {path}"
        for flag, path in required.items()
        if path and not is_remote_path(path) and not Path(path).exists()
    ]
    if missing:
        parser.error(
            "Missing local AlphaGenome reference file(s):\n  "
            + "\n  ".join(missing)
            + "\nDownload them into data/alphagenome_reference/hg38 or override the paths."
        )


def get_fasta_extractor(client, args):
    """Return a FASTA extractor from the model, --fasta-path, or AlphaGenome default URL."""
    try:
        return client._get_fasta_extractor(Organism.HOMO_SAPIENS)
    except Exception as model_exc:
        fasta_path = args.fasta_path or DEFAULT_FASTA_URL
        try:
            from alphagenome_research.io import fasta as local_fasta
        except ModuleNotFoundError as import_exc:
            raise RuntimeError(
                "The local AlphaGenome model has no FASTA extractor, and "
                "alphagenome_research.io.fasta could not be imported. Sync the "
                "updated helper script or pass --fasta-path to a local GRCh38 FASTA."
            ) from import_exc
        try:
            extractor = local_fasta.FastaExtractor(fasta_path)
        except Exception as fasta_exc:
            raise RuntimeError(
                "Could not create a FASTA extractor for patient-level WT sequence "
                f"extraction from {fasta_path!r}. Pass --fasta-path to a local "
                "GRCh38 FASTA if the default remote FASTA is unavailable."
            ) from fasta_exc
        print(
            "Warning: local model had no FASTA extractor; using "
            f"{fasta_path!r} for WT sequence extraction.",
            file=sys.stderr,
        )
        return extractor


def read_patient_ids(args, patient_cancer_map: dict[str, str]) -> list[str]:
    patients = []
    if args.patient_list:
        with Path(args.patient_list).open() as handle:
            patients.extend(line.strip() for line in handle if line.strip())
    if args.patients:
        patients.extend(args.patients)
    if not patients:
        patients = sorted(patient_cancer_map)
    return list(dict.fromkeys(patients))


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


def load_patient_variants(patient_id: str) -> pd.DataFrame:
    path = PATIENT_VARIANTS_DIR / f"{patient_id}.tsv"
    if not path.exists():
        raise FileNotFoundError(f"Missing patient variant file: {path}")
    df = pd.read_csv(path, sep="\t", dtype=str)
    cols_lower = {c.lower(): c for c in df.columns}
    mutation_col = cols_lower.get("mutation") or cols_lower.get("id") or cols_lower.get("variant")
    if mutation_col is None:
        raise ValueError(f"No mutation/id/variant column found in {path}")
    gene_col = cols_lower.get("gene") or cols_lower.get("gene_name")
    rename = {mutation_col: "mutation"}
    if gene_col:
        rename[gene_col] = "variant_gene"
    df = df.rename(columns=rename).dropna(subset=["mutation"]).copy()
    df["mutation"] = df["mutation"].map(normalise_mutation)
    return df


def load_driver_genes(cancer_type: str, driver_gene_file: str | None = None) -> pd.DataFrame:
    cache_key = (cancer_type, str(driver_gene_file))
    if cache_key in DRIVER_GENE_CACHE:
        return DRIVER_GENE_CACHE[cache_key].copy()
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
    DRIVER_GENE_CACHE[cache_key] = df
    return df.copy()


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


def load_cancer_tracks(cancer_type: str, excluded_output_types: set[str]) -> pd.DataFrame:
    cache_key = (cancer_type, tuple(sorted(excluded_output_types)))
    if cache_key in CANCER_TRACK_CACHE:
        return CANCER_TRACK_CACHE[cache_key].copy()
    tracks = pd.read_csv(TRACK_MAPPING, sep="\t", dtype=str).fillna("")
    tracks = tracks[tracks["cancer_type"] == cancer_type].copy()
    if tracks.empty:
        CANCER_TRACK_CACHE[cache_key] = tracks
        return tracks.copy()
    tracks["output_key"] = tracks["output_type"].map(clean_output_type)
    tracks = tracks[~tracks["output_key"].isin(excluded_output_types)].copy()
    tracks["normalized_strand"] = tracks["strand"].map(normalized_track_strand)
    CANCER_TRACK_CACHE[cache_key] = tracks
    return tracks.copy()


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


def selected_ontology_terms(tracks: pd.DataFrame) -> list[str] | None:
    if "ontology_curie" not in tracks:
        return None
    terms = sorted(set(tracks["ontology_curie"].dropna().astype(str)) - {""})
    return terms or None


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


def mutations_in_window(patient_variants: pd.DataFrame, interval: genome.Interval) -> list[str]:
    hits = []
    seq_chrom = normalise_chrom(interval.chromosome)
    for mutation in patient_variants["mutation"]:
        try:
            chrom, start0, end0, _, _ = parse_mutation_interval(mutation)
        except Exception:
            continue
        if chrom == seq_chrom and not (end0 <= interval.start or start0 >= interval.end):
            hits.append(normalise_mutation(mutation))
    return sorted(set(hits))


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


def wt_cache_path(
    cache_dir: Path,
    cancer_type: str,
    gene: str,
    interval: genome.Interval,
    tracks: pd.DataFrame,
    args,
) -> Path:
    track_signature = tracks[["output_key", "track_name", "ontology_curie", "selected_strand"]].sort_values(
        ["output_key", "track_name", "ontology_curie", "selected_strand"]
    ).to_csv(index=False)
    key = "|".join([
        cancer_type,
        gene,
        str(interval.chromosome),
        str(interval.start),
        str(interval.end),
        str(args.model_version),
        str(args.checkpoint_path),
        WT_CACHE_VERSION,
        track_signature,
    ])
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    return cache_dir / cancer_type / f"{safe_patient_filename(gene)}_{digest}.npz"


def tissue_filtered_output(output, tracks: pd.DataFrame):
    filtered = {}
    for output_key, output_tracks in tracks.groupby("output_key", sort=True):
        output_type = alphagenome_output_type(output_key)
        track_values_for_output = filter_track_data_to_curated(
            output.get(output_type), output_tracks
        )
        if track_values_for_output is not None:
            filtered[output_type] = track_values_for_output
    return filtered


def save_wt_cache(path: Path, wt_output, tracks: pd.DataFrame) -> None:
    arrays = {}
    for output_type, wt_track_data in tissue_filtered_output(wt_output, tracks).items():
        output_key = output_type.name
        arrays[f"{output_key}__values"] = track_values(wt_track_data).astype(np.float32)
        arrays[f"{output_key}__resolution"] = np.asarray(wt_track_data.resolution)
        arrays[f"{output_key}__metadata"] = np.asarray(
            wt_track_data.metadata.to_json(orient="split")
        )
    if arrays:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **arrays)


def load_wt_cache(path: Path, interval: genome.Interval, output_types):
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as cached:
            output = {}
            for key in cached.files:
                if not key.endswith("__values"):
                    continue
                output_key = key.removesuffix("__values")
                metadata = pd.read_json(
                    StringIO(str(cached[f"{output_key}__metadata"].item())), orient="split"
                )
                output[alphagenome_output_type(output_key)] = track_data.TrackData(
                    values=cached[key],
                    resolution=int(cached[f"{output_key}__resolution"].item()),
                    metadata=metadata,
                    interval=interval,
                )
        if set(output) != set(output_types):
            return None
        return output
    except Exception as exc:
        print(f"Warning: ignoring invalid WT cache {path}: {exc}", file=sys.stderr)
        return None


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


def slice_track_to_score_interval(track_data, score_interval: genome.Interval):
    if track_data is None:
        return None
    try:
        return track_data.slice_by_interval(score_interval, match_resolution=True)
    except Exception:
        return track_data


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


def patient_output_paths(output_dir: Path, patient_id: str) -> tuple[Path, Path]:
    patient_dir = output_dir / PATIENT_OUTPUT_DIR_NAME
    patient_dir.mkdir(parents=True, exist_ok=True)
    patient_name = safe_patient_filename(patient_id)
    return (
        patient_dir / f"{patient_name}.csv",
        patient_dir / f"{patient_name}{COMPLETED_SUFFIX}",
    )


def progress(iterable, total: int, desc: str, enabled: bool):
    if not enabled or tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, unit="gene")


def score_patient(client, fasta_extractor, patient_id: str, cancer_type: str, args, output_dir: Path) -> None:
    output_path, completed_path = patient_output_paths(output_dir, patient_id)
    if args.force:
        for path in (output_path, completed_path):
            if path.exists():
                path.unlink()

    patient_variants = load_patient_variants(patient_id)
    driver_genes = load_driver_genes(cancer_type, args.driver_gene_file)
    if args.genes:
        driver_genes = driver_genes[driver_genes["gene"].isin(args.genes)].copy()
    if args.max_genes:
        driver_genes = driver_genes.head(args.max_genes).copy()

    gene_to_strand = dict(zip(driver_genes["gene"], driver_genes["strand"], strict=False))
    cancer_tracks = load_cancer_tracks(cancer_type, set(args.excluded_output_types))
    if cancer_tracks.empty:
        print(f"[{patient_id}] No curated AlphaGenome tracks for {cancer_type}, skipping.", file=sys.stderr)
        return

    completed = load_completed(completed_path)
    genes = driver_genes["gene"].dropna().astype(str).tolist()
    gene_rows = {str(row["gene"]): row for _, row in driver_genes.iterrows()}
    print(f"[{patient_id}] {cancer_type}: {len(genes)} candidate genes, {len(patient_variants)} variants")

    rows_written = 0
    completed_now = []
    pending_rows = []
    for gene in progress(genes, total=len(genes), desc=f"{patient_id}", enabled=args.show_progress):
        if (patient_id, gene) in completed:
            continue

        tracks = choose_tracks_for_gene(cancer_tracks, gene, gene_to_strand, args.max_tracks_per_output)
        if tracks.empty:
            completed_now.append(gene)
            continue

        interval = gene_interval(driver_genes, gene, args.sequence_length)
        score_interval = select_score_interval(interval, gene_rows[gene], args.score_region)
        mutations = mutations_in_window(patient_variants, interval)
        if not mutations and not args.include_unmutated_genes:
            completed_now.append(gene)
            continue

        output_types = [alphagenome_output_type(key) for key in sorted(tracks["output_key"].unique())]
        ontology_terms = selected_ontology_terms(tracks) if args.preselect_tracks else None
        try:
            wt_seq = fasta_extractor.extract(interval)
            (
                mut_seq,
                n_applied,
                ref_to_alt,
                inserted_alt_offsets,
                inserted_ref_anchors,
            ) = apply_mutations(wt_seq, interval, mutations)
            if mutations and n_applied == 0:
                print(f"  [{patient_id} {gene}] No mutations applied after reference checks.", file=sys.stderr)

            cache_path = wt_cache_path(
                args.wt_cache_dir, cancer_type, gene, interval, tracks, args
            )
            wt_output = args.wt_memory_cache.get(cache_path)
            if wt_output is None and not args.force_wt_cache:
                wt_output = load_wt_cache(cache_path, interval, output_types)
            if wt_output is None:
                wt_output = client.predict_sequence(
                    wt_seq,
                    organism=Organism.HOMO_SAPIENS,
                    requested_outputs=output_types,
                    ontology_terms=ontology_terms,
                    interval=interval,
                )
                wt_output = tissue_filtered_output(wt_output, tracks)
                save_wt_cache(cache_path, wt_output, tracks)
            args.wt_memory_cache[cache_path] = wt_output
            mut_output = client.predict_sequence(
                mut_seq,
                organism=Organism.HOMO_SAPIENS,
                requested_outputs=output_types,
                ontology_terms=ontology_terms,
                interval=interval,
            )
            rows = summarize_delta(
                patient_id=patient_id,
                cancer_type=cancer_type,
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
                aggregation_types=args.aggregation_types,
                score_region=args.score_region,
            )
        except Exception as exc:
            print(f"  Error scoring {patient_id} {gene}: {exc}", file=sys.stderr)
            continue

        pending_rows.extend(rows)
        completed_now.append(gene)
        if len(completed_now) >= args.batch_size:
            rows_written += append_rows(output_path, pending_rows)
            append_completed(completed_path, patient_id, completed_now)
            completed_now = []
            pending_rows = []

    rows_written += append_rows(output_path, pending_rows)
    append_completed(completed_path, patient_id, completed_now)
    print(f"[{patient_id}] wrote {rows_written:,} score rows -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Patient-level tissue-specific AlphaGenome track-delta scoring.")
    parser.add_argument("--patients", nargs="+", default=DEFAULT_PATIENTS, help="Patient IDs to process.")
    parser.add_argument("--patient-list", default=DEFAULT_PATIENT_LIST, help="Optional text file with one patient ID per line.")
    parser.add_argument("--genes", nargs="+", default=DEFAULT_GENES, help="Optional gene subset.")
    parser.add_argument(
        "--driver-gene-file",
        default=str(DEFAULT_DRIVER_GENE_FILE),
        help=f"Driver gene coordinate TSV (default: {DEFAULT_DRIVER_GENE_FILE}).",
    )
    parser.add_argument("--max-genes", type=int, default=DEFAULT_MAX_GENES, help="Optional max genes per patient.")
    parser.add_argument("--include-unmutated-genes", action="store_true", help="Also score genes with no patient mutations in the interval.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}).")
    parser.add_argument(
        "--wt-cache-dir",
        help="Directory for reusable tissue-filtered WT predictions (default: OUTPUT_DIR/wt_cache).",
    )
    parser.add_argument(
        "--force-wt-cache",
        action="store_true",
        help="Recompute and replace cached WT predictions.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Patient-gene results to buffer before writing scores and completions.",
    )
    parser.add_argument("--max-tracks-per-output", type=int, default=DEFAULT_MAX_TRACKS_PER_OUTPUT)
    parser.add_argument(
        "--preselect-tracks",
        action="store_true",
        help="Request selected tissue ontology terms from the model; benchmark before enabling.",
    )
    parser.add_argument("--excluded-output-types", nargs="+", default=DEFAULT_EXCLUDED_OUTPUT_TYPES)
    parser.add_argument(
        "--aggregation-types",
        nargs="+",
        default=DEFAULT_AGGREGATION_TYPES,
        choices=[item.name for item in variant_scorers.AggregationType],
        help=(
            "AlphaGenome-style aggregation types to apply to WT and patient-mutated tracks "
            f"(default: {' '.join(DEFAULT_AGGREGATION_TYPES)})."
        ),
    )
    parser.add_argument(
        "--score-region",
        choices=["gene", "full_interval"],
        default=DEFAULT_SCORE_REGION,
        help="Region used for scorer aggregation inside the AlphaGenome interval (default: %(default)s).",
    )
    parser.add_argument("--sequence-length", choices=sorted(SEQUENCE_LENGTHS), default=DEFAULT_SEQUENCE_LENGTH_NAME)
    parser.add_argument("--force", action="store_true", default=DEFAULT_FORCE)
    parser.add_argument("--no-progress", action="store_false", dest="show_progress", default=DEFAULT_SHOW_PROGRESS)

    parser.add_argument("--weights-source", choices=["checkpoint", "huggingface", "kaggle"], default=DEFAULT_WEIGHTS_SOURCE)
    parser.add_argument("--checkpoint-path", default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--model-version", default=DEFAULT_MODEL_VERSION)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--allow-cpu", action="store_true", default=DEFAULT_ALLOW_CPU)
    parser.add_argument("--fasta-path", default=DEFAULT_FASTA_PATH)
    parser.add_argument("--gtf-feather-path", default=DEFAULT_GTF_FEATHER_PATH)
    parser.add_argument("--pas-feather-path", default=DEFAULT_PAS_FEATHER_PATH)
    parser.add_argument("--splice-site-starts-feather-path", default=DEFAULT_SPLICE_SITE_STARTS_FEATHER_PATH)
    parser.add_argument("--splice-site-ends-feather-path", default=DEFAULT_SPLICE_SITE_ENDS_FEATHER_PATH)
    parser.add_argument("--calibration-path", default=DEFAULT_CALIBRATION_PATH)
    parser.add_argument("--quiet-model-loading", action="store_false", dest="verbose_model_loading", default=DEFAULT_VERBOSE_MODEL_LOADING)
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")
    if args.max_tracks_per_output < 1:
        parser.error("--max-tracks-per-output must be positive.")
    validate_reference_files(args, parser)
    args.sequence_length = SEQUENCE_LENGTHS[args.sequence_length]
    args.excluded_output_types = [clean_output_type(value) for value in args.excluded_output_types]
    args.aggregation_types = parse_aggregation_types(args.aggregation_types)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    args.wt_cache_dir = Path(args.wt_cache_dir) if args.wt_cache_dir else output_dir / WT_CACHE_DIR_NAME
    args.wt_cache_dir.mkdir(parents=True, exist_ok=True)
    args.wt_memory_cache = {}

    patient_cancer_map = load_patient_cancer_map(PATIENT_CANCER_MAP)
    patients = read_patient_ids(args, patient_cancer_map)
    unknown = [patient for patient in patients if patient not in patient_cancer_map]
    if unknown:
        print(f"Warning: patients not found in cancer map: {unknown}", file=sys.stderr)
    patients = [patient for patient in patients if patient in patient_cancer_map]
    if not patients:
        print("No patients to process.", file=sys.stderr)
        sys.exit(1)

    print(f"Patients to process: {len(patients)}")
    print(f"Patient outputs: {output_dir / PATIENT_OUTPUT_DIR_NAME}")
    start = time.monotonic()
    client = create_local_model(args)
    fasta_extractor = get_fasta_extractor(client, args)

    for patient_id in patients:
        score_patient(
            client=client,
            fasta_extractor=fasta_extractor,
            patient_id=patient_id,
            cancer_type=patient_cancer_map[patient_id],
            args=args,
            output_dir=output_dir,
        )

    print(f"Done in {time.monotonic() - start:.1f}s.")


if __name__ == "__main__":
    main()
