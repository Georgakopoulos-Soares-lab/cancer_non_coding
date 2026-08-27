#!/usr/bin/env python
"""
Curate AlphaGenome target tracks for cancer tissue types.

The cancer tissue labels come from metadata/tcga_cancer_tissue_cell_lines.tsv.
The AlphaGenome track metadata comes from metadata/alphagenome_metadata.csv.
By default, all AlphaGenome output types are considered. Matching prefers tissue
biosamples and falls back to non-cancer primary/in-vitro cell biosamples when no
tissue-level tracks are available for a tissue.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from curate_borzoi_rna_tissue_tracks import (
    ALWAYS_EXCLUDE_PATTERNS,
    CELL_FALLBACK_EXCLUDE_PATTERNS,
    CELL_FALLBACK_PATTERNS,
    TISSUE_PATTERNS,
    keyword_patterns,
    load_cancer_tissues,
    matching_patterns,
)


DEFAULT_CANCER_TISSUES = "metadata/tcga_cancer_tissue_cell_lines.tsv"
DEFAULT_ALPHAGENOME_METADATA = "metadata/alphagenome_metadata.csv"
DEFAULT_OUT = "metadata/cancer_tissue_alphagenome_tracks.tsv"
DEFAULT_TISSUE_OUT = "metadata/tissue_alphagenome_tracks.tsv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Curate AlphaGenome tracks for cancer tissue types.")
    parser.add_argument("--cancer-tissues", default=DEFAULT_CANCER_TISSUES)
    parser.add_argument("--alphagenome-metadata", default=DEFAULT_ALPHAGENOME_METADATA)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--tissue-out", default=DEFAULT_TISSUE_OUT)
    parser.add_argument(
        "--output-types",
        nargs="*",
        default=None,
        help=(
            "Optional AlphaGenome output_type values to include. "
            "Default: include all output types. Example: --output-types OutputType.RNA_SEQ OutputType.CAGE"
        ),
    )
    return parser.parse_args()


def load_alphagenome_tracks(path: str, output_types: set[str] | None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for target_index, row in enumerate(reader):
            if output_types is not None and row.get("output_type") not in output_types:
                continue
            row = dict(row)
            row["target_index"] = str(target_index)
            rows.append(row)
    return rows


def target_text(target: dict[str, str]) -> str:
    return " ".join(
        str(target.get(key, "") or "")
        for key in [
            "name",
            "Assay title",
            "ontology_curie",
            "biosample_name",
            "biosample_type",
            "data_source",
            "output_type",
            "gtex_tissue",
        ]
    )


def is_true(value: str) -> bool:
    return str(value or "").strip().lower() in {"true", "1", "yes"}


def is_base_track(target: dict[str, str]) -> bool:
    text = target_text(target)
    if is_true(target.get("genetically_modified", "")):
        return False
    if matching_patterns(text, ALWAYS_EXCLUDE_PATTERNS):
        return False
    return True


def is_tissue_track(target: dict[str, str]) -> bool:
    if not is_base_track(target):
        return False
    return str(target.get("biosample_type", "")).strip().lower() == "tissue"


def is_cell_fallback_track(target: dict[str, str]) -> bool:
    if not is_base_track(target):
        return False
    biosample_type = str(target.get("biosample_type", "")).strip().lower()
    if biosample_type not in {"primary_cell", "in_vitro_differentiated_cells", "organoid"}:
        return False
    text = target_text(target)
    if matching_patterns(text, CELL_FALLBACK_EXCLUDE_PATTERNS):
        return False
    return bool(matching_patterns(text, CELL_FALLBACK_PATTERNS))


def matching_target_rows(
    targets: list[dict[str, str]],
    patterns: list[str],
    *,
    allow_cells: bool,
) -> list[tuple[dict[str, str], list[str]]]:
    rows = []
    for target in targets:
        if allow_cells:
            if not is_cell_fallback_track(target):
                continue
        elif not is_tissue_track(target):
            continue
        matched = matching_patterns(target_text(target), patterns)
        if matched:
            rows.append((target, matched))
    return rows


def track_sort_key(item: tuple[dict[str, str], list[str]]) -> tuple[int, int, str, str]:
    target, _matched = item
    assay = str(target.get("Assay title", "")).lower()
    has_gtex = 0 if target.get("gtex_tissue") else 1
    assay_rank = 0 if "polya" in assay or "polya plus" in assay else 1
    return (has_gtex, assay_rank, str(target.get("biosample_name", "")), str(target.get("name", "")))


def make_output_row(cancer_row: dict[str, str], target: dict[str, str], matched: list[str]) -> dict[str, str]:
    return {
        "cancer_type": cancer_row["cancer_type"],
        "tissue": cancer_row["tissue"],
        "target_index": target["target_index"],
        "track_name": target.get("name", ""),
        "strand": target.get("strand", ""),
        "assay_title": target.get("Assay title", ""),
        "output_type": target.get("output_type", ""),
        "ontology_curie": target.get("ontology_curie", ""),
        "biosample_name": target.get("biosample_name", ""),
        "biosample_type": target.get("biosample_type", ""),
        "biosample_life_stage": target.get("biosample_life_stage", ""),
        "data_source": target.get("data_source", ""),
        "gtex_tissue": target.get("gtex_tissue", ""),
        "histone_mark": target.get("histone_mark", ""),
        "transcription_factor": target.get("transcription_factor", ""),
        "nonzero_mean": target.get("nonzero_mean", ""),
        "matched_patterns": ";".join(matched),
    }


def write_tsv(path: str, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    cancer_rows = load_cancer_tissues(args.cancer_tissues)
    output_types = set(args.output_types) if args.output_types else None
    targets = load_alphagenome_tracks(args.alphagenome_metadata, output_types)

    missing_tissues = sorted({row["tissue"] for row in cancer_rows} - set(TISSUE_PATTERNS))
    if missing_tissues:
        raise ValueError(f"No curation patterns defined for tissues: {missing_tissues}")

    out_rows: list[dict[str, str]] = []
    tissue_rows: list[dict[str, str]] = []
    seen_tissue_tracks: set[tuple[str, str]] = set()
    fallback_tissues: set[str] = set()

    for cancer_row in cancer_rows:
        tissue = cancer_row["tissue"]
        patterns = [*TISSUE_PATTERNS[tissue], *keyword_patterns(cancer_row.get("keyword", ""))]
        matched_targets = matching_target_rows(targets, patterns, allow_cells=False)
        if not matched_targets:
            fallback_tissues.add(tissue)
            matched_targets = matching_target_rows(targets, patterns, allow_cells=True)

        for target, matched in sorted(matched_targets, key=track_sort_key):
            row = make_output_row(cancer_row, target, matched)
            tissue_key = (tissue, target["target_index"])
            if tissue_key not in seen_tissue_tracks:
                seen_tissue_tracks.add(tissue_key)
                tissue_rows.append({k: v for k, v in row.items() if k != "cancer_type"})
            out_rows.append(row)

    fieldnames = [
        "cancer_type",
        "tissue",
        "target_index",
        "track_name",
        "strand",
        "assay_title",
        "output_type",
        "ontology_curie",
        "biosample_name",
        "biosample_type",
        "biosample_life_stage",
        "data_source",
        "gtex_tissue",
        "histone_mark",
        "transcription_factor",
        "nonzero_mean",
        "matched_patterns",
    ]
    tissue_fieldnames = [field for field in fieldnames if field != "cancer_type"]
    write_tsv(args.out, out_rows, fieldnames)
    write_tsv(args.tissue_out, tissue_rows, tissue_fieldnames)

    tissue_counts: dict[str, int] = {}
    for row in tissue_rows:
        tissue_counts[row["tissue"]] = tissue_counts.get(row["tissue"], 0) + 1
    cancer_counts: dict[str, int] = {}
    for row in out_rows:
        cancer_counts[row["cancer_type"]] = cancer_counts.get(row["cancer_type"], 0) + 1
    missing_cancers = [
        (row["cancer_type"], row["tissue"])
        for row in cancer_rows
        if cancer_counts.get(row["cancer_type"], 0) == 0
    ]

    if output_types is None:
        loaded_output_types = sorted({row.get("output_type", "") for row in targets})
        print(f"Loaded {len(targets)} AlphaGenome track(s) across {len(loaded_output_types)} output type(s)")
        for output_type in loaded_output_types:
            n = sum(1 for row in targets if row.get("output_type", "") == output_type)
            print(f"{output_type}\t{n}")
    else:
        print(f"Loaded {len(targets)} AlphaGenome track(s) with output types: {', '.join(args.output_types)}")
    print(f"Wrote {len(out_rows)} cancer-type/tissue/track mappings to {args.out}")
    print(f"Wrote {len(tissue_rows)} tissue/track mappings to {args.tissue_out}")
    for tissue in sorted(tissue_counts):
        print(f"{tissue}\t{tissue_counts[tissue]}")
    if fallback_tissues:
        print("[INFO] Used normal cell-track fallback for tissues with no tissue-level matches:")
        for tissue in sorted(fallback_tissues):
            print(f"  {tissue}")
    if missing_cancers:
        print("[WARN] Cancer types with no curated AlphaGenome tracks:")
        for cancer_type, tissue in missing_cancers:
            print(f"  {cancer_type}\t{tissue}")


if __name__ == "__main__":
    main()
