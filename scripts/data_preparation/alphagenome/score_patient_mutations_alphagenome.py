#!/usr/bin/env python
"""Score one patient's driver-gene mutations with AlphaGenome.

For a patient ID, this script:
  1. Resolves the patient's TCGA cancer type from
     data/TCGA/tcga_patient_variants_by_cancer and caches the mapping.
  2. Loads the cancer-specific driver genes from data/driver_genes_coords/{CANCER}.tsv.
  3. Keeps only this patient's mutations in those driver genes.
  4. Scores variants with AlphaGenome recommended variant scorers.
  5. Keeps only tracks curated for the patient's cancer type in
     metadata/cancer_tissue_alphagenome_tracks.tsv.

The scorer choice follows AlphaGenome's variant-scoring documentation:
recommended scorers cover RNA-seq, polyadenylation, CAGE/PRO-cap, ATAC/DNase,
ChIP-TF, ChIP-histone, splice-site, splice-site-usage, splice-junction, contact
map, and active-allele variants where supported.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alphagenome.data import genome
from alphagenome.models import dna_client, variant_scorers as vs_lib
from alphagenome.models.dna_model import Organism


PROJECT_DIR = Path(__file__).resolve().parents[3]
TCGA_BY_CANCER_DIR = PROJECT_DIR / "data" / "TCGA" / "tcga_patient_variants_by_cancer"
PATIENT_CANCER_MAP = PROJECT_DIR / "data" / "tcga_patient_cancer_map.json"
DRIVER_GENE_DIR = PROJECT_DIR / "data" / "driver_genes_coords"
TRACK_MAPPING = PROJECT_DIR / "metadata" / "cancer_tissue_alphagenome_tracks.tsv"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data" / "alphagenome_scores" / "patient_driver_mutations"
SEQUENCE_LENGTH = getattr(dna_client, "SEQUENCE_LENGTH_1MB", dna_client.SEQUENCE_LENGTH_100KB)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score patient driver-gene mutations with AlphaGenome.")
    parser.add_argument("patient_id", help="TCGA patient ID, e.g. TCGA-02-0016")
    parser.add_argument("--api-key", default=".alphagenome_api_key", help="File containing AlphaGenome API key.")
    parser.add_argument("--server", default=None, help="Optional AlphaGenome gRPC server address.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-file", default=None, help="Override output CSV path.")
    parser.add_argument("--patient-cancer-map", default=str(PATIENT_CANCER_MAP))
    parser.add_argument("--variants-dir", default=str(TCGA_BY_CANCER_DIR))
    parser.add_argument("--driver-gene-dir", default=str(DRIVER_GENE_DIR))
    parser.add_argument("--track-mapping", default=str(TRACK_MAPPING))
    parser.add_argument("--cancer-type", default=None, help="Override inferred cancer type.")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--max-mutations", type=int, default=None, help="Limit driver mutations for testing.")
    parser.add_argument(
        "--genes",
        nargs="+",
        default=None,
        help=(
            "Optional gene symbol(s) to score. Values may be space-separated "
            "or comma-separated."
        ),
    )
    parser.add_argument(
        "--scorers",
        nargs="*",
        default=None,
        help="Optional recommended scorer keys to use. Default: all recommended scorers whose output type has curated tracks.",
    )
    parser.add_argument("--force-map", action="store_true", help="Rebuild patient-cancer map even if cache exists.")
    return parser.parse_args()


def build_patient_cancer_map(variants_dir: Path, out_path: Path) -> dict[str, str]:
    """Build patient -> cancer type map from TCGA *_tmb.tsv files only."""
    mapping: dict[str, str] = {}
    for path in sorted(variants_dir.glob("*_tmb.tsv")):
        cancer_type = path.name.removesuffix("_tmb.tsv")
        try:
            df = pd.read_csv(path, sep="\t", dtype=str, usecols=["bcr_patient_barcode"])
        except Exception as exc:
            print(f"[WARN] Could not read {path}: {exc}", file=sys.stderr)
            continue
        for patient_id in df["bcr_patient_barcode"].dropna().astype(str):
            mapping[patient_id] = cancer_type

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(mapping, indent=2, sort_keys=True))
    print(f"Saved patient-cancer map for {len(mapping):,} patients to {out_path}")
    return mapping


def load_patient_cancer_map(variants_dir: Path, map_path: Path, force: bool = False) -> dict[str, str]:
    if map_path.exists() and not force:
        return json.loads(map_path.read_text())
    return build_patient_cancer_map(variants_dir, map_path)


def load_patient_mutations(patient_id: str, cancer_type: str, variants_dir: Path) -> pd.DataFrame:
    path = variants_dir / f"{cancer_type}_variants.tsv"
    if not path.exists():
        raise FileNotFoundError(f"Variant file not found for cancer type {cancer_type}: {path}")

    rows: list[dict[str, str]] = []
    seen_patient = False
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"bcr_patient_barcode", "mutation", "gene_name"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing required column(s): {sorted(missing)}")
        for row in reader:
            current_patient = row["bcr_patient_barcode"]
            if current_patient == patient_id:
                seen_patient = True
                rows.append(
                    {
                        "bcr_patient_barcode": current_patient,
                        "mutation": row["mutation"],
                        "gene_name": row["gene_name"],
                    }
                )
            elif seen_patient:
                # TCGA by-cancer variant files are grouped by patient; stop once
                # the requested patient's contiguous block is complete.
                break

    return pd.DataFrame(rows, columns=["bcr_patient_barcode", "mutation", "gene_name"])


def load_driver_genes(cancer_type: str, driver_gene_dir: Path) -> tuple[set[str], dict[str, str]]:
    path = driver_gene_dir / f"{cancer_type}.tsv"
    if not path.exists():
        print(f"[WARN] Missing {path}; falling back to Pancancer.tsv", file=sys.stderr)
        path = driver_gene_dir / "Pancancer.tsv"
    if not path.exists():
        raise FileNotFoundError(f"Driver gene file not found: {path}")
    df = pd.read_csv(path, sep="\t", dtype=str)
    if "gene" not in df.columns:
        raise ValueError(f"{path} is missing required column: gene")
    genes = set(df["gene"].dropna().astype(str))
    gene_to_strand = {}
    if "strand" in df.columns:
        gene_to_strand = dict(
            zip(
                df["gene"].fillna("").astype(str),
                df["strand"].fillna("").astype(str),
            )
        )
    return genes, gene_to_strand


def load_requested_gene_fallbacks(
    requested_genes: set[str],
    driver_gene_dir: Path,
) -> tuple[set[str], dict[str, str]]:
    """Load requested genes from broader driver-coordinate files."""
    genes: set[str] = set()
    gene_to_strand: dict[str, str] = {}
    for filename in [
        "Pancancer_1pc.tsv",
        "Pancancer.tsv",
        "all_driver_genes_hg38.tsv",
    ]:
        path = driver_gene_dir / filename
        if not path.exists():
            continue
        df = pd.read_csv(path, sep="\t", dtype=str)
        if "gene" not in df.columns:
            continue
        df = df[df["gene"].isin(requested_genes)].copy()
        genes.update(df["gene"].dropna().astype(str))
        if "strand" in df.columns:
            gene_to_strand.update(
                zip(
                    df["gene"].fillna("").astype(str),
                    df["strand"].fillna("").astype(str),
                )
            )
    return genes, gene_to_strand


def load_cancer_track_mapping(cancer_type: str, path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"AlphaGenome tissue-track mapping not found: {path}")
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    df = df[df["cancer_type"] == cancer_type].copy()
    if df.empty:
        raise ValueError(f"No AlphaGenome tissue-track rows found for cancer type {cancer_type} in {path}")
    return df


def normalize_output_type(value: Any) -> str:
    text = str(value)
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    return text.upper()


def scorer_output_type(scorer: Any) -> str:
    output_type = normalize_output_type(getattr(scorer, "requested_output", ""))
    if output_type:
        return output_type
    name = str(getattr(scorer, "name", "") or scorer.__class__.__name__).upper()
    if "POLYADENYLATION" in name:
        return "RNA_SEQ"
    if "SPLICE_JUNCTION" in name:
        return "SPLICE_JUNCTIONS"
    if "SPLICE_SITE_USAGE" in name:
        return "SPLICE_SITE_USAGE"
    if "SPLICE_SITE" in name:
        return "SPLICE_SITES"
    if "CONTACT" in name:
        return "CONTACT_MAPS"
    return ""


def recommended_scorers_by_name() -> dict[str, Any]:
    if hasattr(vs_lib, "RECOMMENDED_VARIANT_SCORERS"):
        return {str(k).upper(): v for k, v in dict(vs_lib.RECOMMENDED_VARIANT_SCORERS).items()}

    scorers = vs_lib.get_recommended_scorers(Organism.HOMO_SAPIENS)
    out: dict[str, Any] = {}
    for scorer in scorers:
        name = str(getattr(scorer, "name", "") or scorer.__class__.__name__)
        out[name.upper()] = scorer
    return out


def select_scorers(track_df: pd.DataFrame, requested_names: list[str] | None) -> tuple[list[str], list[Any]]:
    scorer_map = recommended_scorers_by_name()
    if requested_names:
        requested_names = [name.upper() for name in requested_names]
        missing = [name for name in requested_names if name not in scorer_map]
        if missing:
            raise ValueError(f"Unknown recommended scorer(s): {missing}. Available: {sorted(scorer_map)}")
        selected = {name: scorer_map[name] for name in requested_names}
    else:
        curated_output_types = {normalize_output_type(x) for x in track_df["output_type"].unique()}
        selected = {
            name: scorer
            for name, scorer in scorer_map.items()
            if scorer_output_type(scorer) in curated_output_types
        }

    if not selected:
        raise ValueError("No AlphaGenome recommended scorers matched the curated track output types.")
    return list(selected.keys()), list(selected.values())


def parse_mutation(mutation_str: str) -> tuple[str, int, str, str]:
    parts = str(mutation_str).split(":")
    if len(parts) != 4:
        raise ValueError(f"Unexpected mutation format: {mutation_str!r}")
    chrom_raw, coord_str, ref, alt = parts
    chrom = chrom_raw if chrom_raw.startswith("chr") else f"chr{chrom_raw}"
    pos = int(coord_str.split("-")[0])
    return chrom, pos, ref, alt


def make_centered_interval(chrom: str, pos: int, ref: str, seq_length: int = SEQUENCE_LENGTH) -> genome.Interval:
    center = (pos - 1) + len(ref) // 2
    start = max(0, center - seq_length // 2)
    end = start + seq_length
    return genome.Interval(chromosome=chrom, start=start, end=end)


def clean_track_value(value) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def choose_allowed_strand_subset(allowed: pd.DataFrame, desired_strand: str | None) -> pd.DataFrame:
    if "strand" not in allowed.columns or desired_strand not in {"+", "-"}:
        return allowed

    strand_values = allowed["strand"].map(clean_track_value)
    for candidate in [desired_strand, ".", ""]:
        candidate_rows = allowed[strand_values == candidate]
        if not candidate_rows.empty:
            return candidate_rows
    return allowed.iloc[0:0].copy()


def track_allowed_mask(
    var_df: pd.DataFrame,
    allowed_tracks: pd.DataFrame,
    output_type: str,
    desired_strand: str | None = None,
) -> pd.Series:
    allowed = allowed_tracks[allowed_tracks["output_type"].map(normalize_output_type) == output_type]
    allowed = choose_allowed_strand_subset(allowed, desired_strand)
    if allowed.empty:
        return pd.Series(False, index=var_df.index)

    false_mask = pd.Series(False, index=var_df.index)

    # Prefer exact AlphaGenome track-name matching. The previous implementation
    # OR-matched broad metadata fields such as strand, which let unrelated
    # tissues through whenever they shared +, -, or . with an allowed track.
    if "name" in var_df.columns and "track_name" in allowed.columns:
        allowed_names = {clean_track_value(v) for v in allowed["track_name"]}
        allowed_names.discard("")
        if allowed_names:
            mask = var_df["name"].map(clean_track_value).isin(allowed_names)
            if mask.any():
                return mask

    identity_pairs = []
    for var_col, allowed_col in [
        ("ontology_curie", "ontology_curie"),
        ("biosample_name", "biosample_name"),
    ]:
        if var_col in var_df.columns and allowed_col in allowed.columns:
            identity_pairs.append((var_col, allowed_col))

    # Fallback only when we can match on actual tissue identity. Matching on
    # optional fields alone, especially strand, is too permissive.
    if len(identity_pairs) < 2:
        return false_mask

    optional_pairs = []
    for var_col, allowed_col in [
        ("strand", "strand"),
        ("gtex_tissue", "gtex_tissue"),
        ("histone_mark", "histone_mark"),
        ("transcription_factor", "transcription_factor"),
    ]:
        if var_col in var_df.columns and allowed_col in allowed.columns:
            allowed_values = {clean_track_value(v) for v in allowed[allowed_col]}
            var_values = {clean_track_value(v) for v in var_df[var_col]}
            if allowed_values.difference({""}) and var_values.difference({""}):
                optional_pairs.append((var_col, allowed_col))

    pairs = identity_pairs + optional_pairs
    allowed_keys = {
        tuple(clean_track_value(row[allowed_col]) for _, allowed_col in pairs)
        for _, row in allowed.iterrows()
    }

    def row_is_allowed(row) -> bool:
        key = tuple(clean_track_value(row[var_col]) for var_col, _ in pairs)
        return key in allowed_keys

    return var_df.apply(row_is_allowed, axis=1)


def filter_anndata_to_curated_tracks(
    adata,
    allowed_tracks: pd.DataFrame,
    output_type: str,
    desired_strand: str | None = None,
):
    var_df = adata.var
    mask = track_allowed_mask(var_df, allowed_tracks, output_type, desired_strand=desired_strand)
    if mask.any():
        return adata[:, mask.to_numpy()]
    return adata[:, []]


def score_batch(
    client,
    variants: list,
    intervals: list,
    scorer_names: list[str],
    scorers: list,
    mutation_meta: list[dict[str, Any]],
    allowed_tracks: pd.DataFrame,
    max_workers: int,
) -> tuple[list[dict[str, Any]], bool]:
    try:
        all_results = client.score_variants(
            intervals=intervals,
            variants=variants,
            variant_scorers=scorers,
            organism=Organism.HOMO_SAPIENS,
            progress_bar=False,
            max_workers=max_workers,
        )
    except Exception as exc:
        print(f"[ERROR] AlphaGenome batch scoring failed: {exc}", file=sys.stderr)
        return [], False

    rows: list[dict[str, Any]] = []
    for meta, scorer_results in zip(mutation_meta, all_results):
        for scorer_name, scorer, adata in zip(scorer_names, scorers, scorer_results):
            output_type = scorer_output_type(scorer)
            filtered = filter_anndata_to_curated_tracks(
                adata,
                allowed_tracks,
                output_type,
                desired_strand=meta.get("gene_strand"),
            )
            if filtered.n_obs == 0 or filtered.n_vars == 0:
                continue
            scores = np.asarray(filtered.X[0], dtype=np.float32)
            var_df = filtered.var
            for i in range(filtered.n_vars):
                row = dict(meta)
                row["scorer"] = scorer_name
                row["output_type"] = output_type
                row["score"] = float(scores[i])
                for col in [
                    "name",
                    "strand",
                    "ontology_curie",
                    "biosample_name",
                    "biosample_type",
                    "biosample_life_stage",
                    "data_source",
                    "gtex_tissue",
                    "histone_mark",
                    "transcription_factor",
                ]:
                    row[col] = var_df.iloc[i].get(col, np.nan) if col in var_df.columns else np.nan
                rows.append(row)
    return rows, True


def score_batch_with_retry(
    client,
    variants: list,
    intervals: list,
    scorer_names: list[str],
    scorers: list,
    mutation_meta: list[dict[str, Any]],
    allowed_tracks: pd.DataFrame,
    max_workers: int,
) -> list[dict[str, Any]]:
    rows, ok = score_batch(
        client=client,
        variants=variants,
        intervals=intervals,
        scorer_names=scorer_names,
        scorers=scorers,
        mutation_meta=mutation_meta,
        allowed_tracks=allowed_tracks,
        max_workers=max_workers,
    )
    if ok:
        return rows
    if len(variants) <= 1:
        failed = mutation_meta[0].get("mutation", "<unknown>") if mutation_meta else "<unknown>"
        print(f"[WARN] Skipping variant after single-variant AlphaGenome failure: {failed}", file=sys.stderr)
        return []

    mid = len(variants) // 2
    print(
        f"[WARN] Retrying failed batch as two smaller batches: "
        f"{mid} and {len(variants) - mid} variants",
        file=sys.stderr,
    )
    return [
        *score_batch_with_retry(
            client,
            variants[:mid],
            intervals[:mid],
            scorer_names,
            scorers,
            mutation_meta[:mid],
            allowed_tracks,
            max_workers,
        ),
        *score_batch_with_retry(
            client,
            variants[mid:],
            intervals[mid:],
            scorer_names,
            scorers,
            mutation_meta[mid:],
            allowed_tracks,
            max_workers,
        ),
    ]


def main() -> None:
    # Make progress visible even when launched through `conda run` or batch jobs.
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    args = parse_args()
    patient_id = args.patient_id
    variants_dir = Path(args.variants_dir)
    map_path = Path(args.patient_cancer_map)

    patient_map = load_patient_cancer_map(variants_dir, map_path, force=args.force_map)
    cancer_type = args.cancer_type or patient_map.get(patient_id)
    if not cancer_type:
        raise ValueError(f"Could not infer cancer type for {patient_id}; pass --cancer-type.")
    print(f"Patient: {patient_id}")
    print(f"Cancer type: {cancer_type}")

    driver_gene_dir = Path(args.driver_gene_dir)
    driver_genes, gene_to_strand = load_driver_genes(cancer_type, driver_gene_dir)
    requested_genes = None
    if args.genes:
        requested_genes = {
            gene.strip()
            for value in args.genes
            for gene in str(value).split(",")
            if gene.strip()
        }
        cancer_driver_genes = driver_genes.intersection(requested_genes)
        missing_requested_genes = requested_genes.difference(cancer_driver_genes)
        fallback_genes, fallback_strands = load_requested_gene_fallbacks(
            missing_requested_genes,
            driver_gene_dir,
        )
        driver_genes = cancer_driver_genes.union(fallback_genes)
        gene_to_strand.update(fallback_strands)
        if not driver_genes:
            raise ValueError(
                "None of the requested gene(s) are in the driver-gene coordinate files: "
                f"{sorted(requested_genes)}"
            )
        print(f"Requested genes: {', '.join(sorted(requested_genes))}")
        if missing_requested_genes:
            found_fallbacks = sorted(fallback_genes.intersection(missing_requested_genes))
            still_missing = sorted(missing_requested_genes.difference(fallback_genes))
            if found_fallbacks:
                print(
                    "Requested genes found outside the cancer-specific driver set: "
                    f"{', '.join(found_fallbacks)}"
                )
            if still_missing:
                print(
                    "[WARN] Requested genes missing from fallback driver-coordinate files: "
                    f"{', '.join(still_missing)}",
                    file=sys.stderr,
                )
    print(f"Driver genes selected for scoring: {len(driver_genes)}")

    mutations = load_patient_mutations(patient_id, cancer_type, variants_dir)
    if mutations.empty:
        print(f"No variants found for {patient_id} in {cancer_type}.")
        return
    mutations = mutations[mutations["gene_name"].isin(driver_genes)].drop_duplicates(["mutation", "gene_name"])
    print(f"Driver-gene mutations: {len(mutations)} ({mutations['gene_name'].nunique()} genes)")
    if args.max_mutations is not None:
        mutations = mutations.head(args.max_mutations)
        print(f"Testing limit applied: {len(mutations)} mutations")
    if mutations.empty:
        print("No driver-gene mutations to score.")
        return

    allowed_tracks = load_cancer_track_mapping(cancer_type, Path(args.track_mapping))
    print(
        f"Curated AlphaGenome tracks for {cancer_type}: {len(allowed_tracks)} rows, "
        f"{allowed_tracks['output_type'].nunique()} output types"
    )

    scorer_names, scorers = select_scorers(allowed_tracks, args.scorers)
    print(f"Selected recommended scorers ({len(scorers)}): {', '.join(scorer_names)}")

    variants, intervals, mutation_meta = [], [], []
    skipped = 0
    for _, row in mutations.iterrows():
        mut_str = row["mutation"]
        try:
            chrom, pos, ref, alt = parse_mutation(mut_str)
            variants.append(
                genome.Variant(
                    chromosome=chrom,
                    position=pos,
                    reference_bases=ref,
                    alternate_bases=alt,
                    name=mut_str,
                )
            )
            intervals.append(make_centered_interval(chrom, pos, ref))
            mutation_meta.append(
                {
                    "patient_id": patient_id,
                    "cancer_type": cancer_type,
                    "mutation": mut_str,
                    "gene": row.get("gene_name", ""),
                    "gene_strand": gene_to_strand.get(str(row.get("gene_name", "")), ""),
                    "chr": chrom,
                    "pos": pos,
                    "ref": ref,
                    "alt": alt,
                }
            )
        except Exception as exc:
            print(f"[WARN] Skipping {mut_str!r}: {exc}", file=sys.stderr)
            skipped += 1

    if skipped:
        print(f"Skipped {skipped} unparseable mutations.")
    if not variants:
        print("No valid variants to score.")
        return

    api_key_path = Path(args.api_key)
    if not api_key_path.is_file():
        print(f"ERROR: API key file not found: {api_key_path}", file=sys.stderr)
        sys.exit(1)
    api_key = api_key_path.read_text().strip()
    print("Connecting to AlphaGenome server...")
    client = dna_client.create(api_key=api_key, address=args.server)

    all_rows: list[dict[str, Any]] = []
    n_batches = (len(variants) + args.batch_size - 1) // args.batch_size
    for batch_idx in range(n_batches):
        lo = batch_idx * args.batch_size
        hi = min(lo + args.batch_size, len(variants))
        print(f"Batch {batch_idx + 1}/{n_batches}: variants {lo + 1}-{hi} of {len(variants)}")
        rows = score_batch_with_retry(
            client=client,
            variants=variants[lo:hi],
            intervals=intervals[lo:hi],
            scorer_names=scorer_names,
            scorers=scorers,
            mutation_meta=mutation_meta[lo:hi],
            allowed_tracks=allowed_tracks,
            max_workers=args.max_workers,
        )
        all_rows.extend(rows)
        print(f"  -> {len(rows)} rows (total {len(all_rows)})")

    if not all_rows:
        print("No scores produced after curated tissue-track filtering.", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.output_file) if args.output_file else Path(args.output_dir) / f"{patient_id}_{cancer_type}_alphagenome_scores.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out_path, index=False)
    print(f"Saved {len(all_rows):,} score rows to {out_path}")


if __name__ == "__main__":
    main()
