#!/usr/bin/env python
"""Predict selected ChIP-TF tracks for TCGA patients.

For every mutated cancer-specific driver-gene interval, the script saves the
reference and patient-mutated predictions for tracks matching both the chosen
cell lines and transcription factors. It scores L2_DIFF and DIFF_MEAN across
the promoter (TSS +/-2 kb) by default.

Example:
    python scripts/data_preparation/alphagenome/predict_patient_chip_tf_tracks.py \
        --cancer-types LIHC \
        --cell-lines HepG2 \
        --transcription-factors HNF4A HNF4G FOXA1 FOXA2 CEBPA CEBPB HNF1A

    python scripts/data_preparation/alphagenome/predict_patient_chip_tf_tracks.py \
        --cancer-types BRCA \
        --cell-lines MCF-7 \
        --transcription-factors CTCF CUX1 EP300 ESR1 FOXA1 FOXM1 GATA3 KLF4 MAX
    python scripts/data_preparation/alphagenome/predict_patient_chip_tf_tracks.py \
        --cancer-types LUAD LUSC \
        --cell-lines A549 \
        --transcription-factors EP300 NFE2L2 RAD21
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from alphagenome.models import dna_output, variant_scorers
from alphagenome.models.dna_model import Organism

# The existing patient-level helpers use sibling imports, so make this
# directory importable when this script is launched with runpy or from a
# notebook rather than directly from this directory.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import score_patient_tissue_tracks_alphagenome_local_cancer_specific as common


PROJECT_DIR = Path(__file__).resolve().parents[3]
TRACK_METADATA = PROJECT_DIR / "metadata" / "alphagenome_metadata.csv"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data" / "alphagenome_scores" / "patient_chip_tf_tracks"
AGGREGATION_TYPES = [
    variant_scorers.AggregationType.L2_DIFF,
    variant_scorers.AggregationType.DIFF_MEAN,
]
PROMOTER_FLANK_BP = 2_000


def load_chip_tf_tracks(cell_lines: list[str], transcription_factors: list[str]) -> pd.DataFrame:
    tracks = pd.read_csv(TRACK_METADATA, dtype=str).fillna("")
    tracks = tracks[
        tracks["output_type"].eq("OutputType.CHIP_TF")
        & tracks["biosample_name"].isin(cell_lines)
        & tracks["transcription_factor"].isin(transcription_factors)
    ].copy()
    if tracks.empty:
        raise ValueError("No CHIP_TF tracks matched the selected cell lines and transcription factors")
    return tracks.assign(
        track_name=tracks["name"],
        output_key="CHIP_TF",
        selected_strand=".",
    ).drop_duplicates("track_name")


def track_path(output_dir: Path, patient_id: str, gene: str) -> Path:
    patient = common.safe_patient_filename(patient_id)
    return output_dir / "tracks" / patient / f"{gene}.npz"


def save_tracks(path: Path, wt_track_data, mut_track_data, interval, metadata: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        ref=common.track_values(wt_track_data),
        alt=common.track_values(mut_track_data),
        chromosome=str(interval.chromosome),
        start=int(interval.start),
        end=int(interval.end),
        resolution=int(getattr(wt_track_data, "resolution", 1) or 1),
        track_name=metadata["name"].astype(str).to_numpy(),
        ontology_curie=metadata["ontology_curie"].astype(str).to_numpy(),
        biosample_name=metadata["biosample_name"].astype(str).to_numpy(),
        transcription_factor=metadata["transcription_factor"].astype(str).to_numpy(),
    )


def select_chip_tf_score_interval(interval, gene_row: pd.Series, score_region: str):
    if score_region != "tss_plus_2kb":
        return common.select_score_interval(interval, gene_row, score_region)

    gene_start = int(gene_row["start"]) - 1
    gene_end = int(gene_row["end"])
    tss = gene_end if str(gene_row.get("strand", "+")) == "-" else gene_start
    return common.genome.Interval(
        chromosome=interval.chromosome,
        start=max(interval.start, tss - PROMOTER_FLANK_BP),
        end=min(interval.end, tss + PROMOTER_FLANK_BP),
    )


def score_patient(client, fasta_extractor, patient_id: str, cancer_type: str, tracks: pd.DataFrame, args, output_dir: Path) -> None:
    score_path, completed_path = common.patient_output_paths(output_dir, patient_id)
    if args.force:
        for path in (score_path, completed_path, track_path(output_dir, patient_id, "" ).parent):
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()

    patient_variants = common.load_patient_variants(patient_id)
    driver_genes = common.load_driver_genes(cancer_type, args.driver_gene_file)
    if args.genes:
        driver_genes = driver_genes[driver_genes["gene"].isin(args.genes)].copy()
    if args.max_genes:
        driver_genes = driver_genes.head(args.max_genes).copy()

    gene_to_strand = dict(zip(driver_genes["gene"], driver_genes["strand"], strict=False))
    gene_rows = {str(row["gene"]): row for _, row in driver_genes.iterrows()}
    completed = common.load_completed(completed_path)
    completed_now = []
    rows_written = 0

    genes = driver_genes["gene"].dropna().astype(str).tolist()
    print(f"[{patient_id}] {cancer_type}: {len(genes)} genes, {len(patient_variants)} variants")
    for gene in common.progress(genes, len(genes), patient_id, args.show_progress):
        if (patient_id, gene) in completed:
            continue

        interval = common.gene_interval(driver_genes, gene, args.sequence_length)
        mutations = common.mutations_in_window(patient_variants, interval)
        if not mutations and not args.include_unmutated_genes:
            completed_now.append(gene)
            continue

        try:
            wt_seq = fasta_extractor.extract(interval)
            mut_seq, n_applied, ref_to_alt, inserted_alt_offsets, inserted_ref_anchors = common.apply_mutations(
                wt_seq, interval, mutations
            )
            if mutations and not n_applied:
                print(f"  [{patient_id} {gene}] no mutations applied after reference checks", file=sys.stderr)

            kwargs = {
                "organism": Organism.HOMO_SAPIENS,
                "requested_outputs": [dna_output.OutputType.CHIP_TF],
                "ontology_terms": None,
                "interval": interval,
            }
            wt_output = client.predict_sequence(wt_seq, **kwargs)
            mut_output = client.predict_sequence(mut_seq, **kwargs)
            wt_track_data = common.filter_track_data_to_curated(
                wt_output.get(dna_output.OutputType.CHIP_TF), tracks
            )
            mut_track_data = common.filter_track_data_to_curated(
                mut_output.get(dna_output.OutputType.CHIP_TF), tracks
            )
            if wt_track_data is None or mut_track_data is None:
                raise ValueError("Selected CHIP_TF tracks were absent from the prediction")

            score_interval = select_chip_tf_score_interval(interval, gene_rows[gene], args.score_region)
            rows = common.summarize_delta(
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
                aggregation_types=AGGREGATION_TYPES,
                score_region=args.score_region,
            )
            rows = [row for row in rows if row["scorer"] == "PATIENT_SINGLE_TRACK_SCORER"]
            save_tracks(track_path(output_dir, patient_id, gene), wt_track_data, mut_track_data, interval, wt_track_data.metadata)
        except Exception as exc:
            print(f"  Error scoring {patient_id} {gene}: {exc}", file=sys.stderr)
            continue

        rows_written += common.append_rows(score_path, rows)
        completed_now.append(gene)
        if len(completed_now) >= args.batch_size:
            common.append_completed(completed_path, patient_id, completed_now)
            completed_now = []

    common.append_completed(completed_path, patient_id, completed_now)
    print(f"[{patient_id}] wrote {rows_written} scores -> {score_path}")


def main():
    parser = argparse.ArgumentParser(description="Predict selected ChIP-TF tracks for TCGA patients.")
    parser.add_argument(
        "--cancer-types",
        nargs="+",
        required=True,
        help="TCGA cancer type. Patients are loaded from data/tcga_patient_cancer_map.json.",
    )
    parser.add_argument("--cell-lines", nargs="+", required=True, help="Cell lines used to select CHIP_TF tracks.")
    parser.add_argument("--transcription-factors", nargs="+", required=True, help="Transcription factors used to select CHIP_TF tracks.")
    parser.add_argument("--genes", nargs="+", help="Optional driver-gene subset.")
    parser.add_argument("--driver-gene-file", help="Optional driver-gene TSV for all selected patients.")
    parser.add_argument("--max-genes", type=int, help="Optional maximum genes per patient.")
    parser.add_argument("--include-unmutated-genes", action="store_true")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--score-region",
        choices=["tss_plus_2kb", "gene", "full_interval"],
        default="tss_plus_2kb",
        help="Region used for score aggregation (default: TSS +/-2 kb).",
    )
    parser.add_argument("--sequence-length", choices=sorted(common.SEQUENCE_LENGTHS), default=common.DEFAULT_SEQUENCE_LENGTH_NAME)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-progress", action="store_false", dest="show_progress", default=True)

    parser.add_argument("--weights-source", choices=["checkpoint", "huggingface", "kaggle"], default=common.DEFAULT_WEIGHTS_SOURCE)
    parser.add_argument("--checkpoint-path", default=common.DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--model-version", default=common.DEFAULT_MODEL_VERSION)
    parser.add_argument("--device", default=common.DEFAULT_DEVICE)
    parser.add_argument("--allow-cpu", action="store_true", default=common.DEFAULT_ALLOW_CPU)
    parser.add_argument("--fasta-path", default=common.DEFAULT_FASTA_PATH)
    parser.add_argument("--gtf-feather-path", default=common.DEFAULT_GTF_FEATHER_PATH)
    parser.add_argument("--pas-feather-path", default=common.DEFAULT_PAS_FEATHER_PATH)
    parser.add_argument("--splice-site-starts-feather-path", default=common.DEFAULT_SPLICE_SITE_STARTS_FEATHER_PATH)
    parser.add_argument("--splice-site-ends-feather-path", default=common.DEFAULT_SPLICE_SITE_ENDS_FEATHER_PATH)
    parser.add_argument("--calibration-path", default=common.DEFAULT_CALIBRATION_PATH)
    parser.add_argument("--quiet-model-loading", action="store_false", dest="verbose_model_loading", default=common.DEFAULT_VERBOSE_MODEL_LOADING)
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    common.validate_reference_files(args, parser)
    args.sequence_length = common.SEQUENCE_LENGTHS[args.sequence_length]

    patient_cancer_map = common.load_patient_cancer_map(common.PATIENT_CANCER_MAP)
    unknown_cancers = sorted(set(args.cancer_types).difference(patient_cancer_map.values()))
    if unknown_cancers:
        parser.error(f"Cancer types not found in {common.PATIENT_CANCER_MAP}: {', '.join(unknown_cancers)}")
    patients = sorted(patient for patient, cancer_type in patient_cancer_map.items() if cancer_type in args.cancer_types)
    if not patients:
        parser.error("No patients were found for the selected cancer types")
    tracks = load_chip_tf_tracks(args.cell_lines, args.transcription_factors)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run_metadata.json").open("w") as handle:
        json.dump({"cancer_types": args.cancer_types, "cell_lines": args.cell_lines, "transcription_factors": args.transcription_factors}, handle, indent=2)

    print(f"Patients to process: {len(patients)}")
    print(f"Selected CHIP_TF tracks: {len(tracks)}")
    start = time.monotonic()
    client = common.create_local_model(args)
    fasta_extractor = common.get_fasta_extractor(client, args)
    for patient_id in patients:
        score_patient(client, fasta_extractor, patient_id, patient_cancer_map[patient_id], tracks, args, output_dir)
    print(f"Done in {time.monotonic() - start:.1f}s")


if __name__ == "__main__":
    main()
