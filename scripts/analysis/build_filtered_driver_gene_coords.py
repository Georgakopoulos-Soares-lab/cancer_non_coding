#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Set


def _read_list(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"List file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        items = [line.strip() for line in f if line.strip()]
    if not items:
        raise ValueError(f"List file is empty: {path}")
    return items


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    base = root / "results" / "driver_gene_overlap"
    parser = argparse.ArgumentParser(
        description=(
            "Filter driver gene coordinate TSVs to a chosen cancer-type list and gene list."
        )
    )
    parser.add_argument(
        "--cancer-types-file",
        type=Path,
        default=base / "max_subset_min_common_5_cancer_types.txt",
        help="Text file with one cancer type per line.",
    )
    parser.add_argument(
        "--genes-file",
        type=Path,
        default=base / "max_subset_min_common_5_common_genes.txt",
        help="Text file with one gene per line.",
    )
    parser.add_argument(
        "--coords-dir",
        type=Path,
        default=root / "data" / "driver_genes_coords",
        help="Directory containing per-cancer coordinate TSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base / "max_subset_min_common_5_filtered_coords",
        help="Output directory for filtered TSVs and summary.",
    )
    return parser.parse_args()


def _read_tsv_rows(path: Path) -> tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")
        if "gene" not in reader.fieldnames:
            raise ValueError(f"Missing 'gene' column in {path}")
        rows = list(reader)
        return reader.fieldnames, rows


def _write_tsv(path: Path, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()

    cancer_types = _read_list(args.cancer_types_file.resolve())
    genes = set(_read_list(args.genes_file.resolve()))

    coords_dir = args.coords_dir.resolve()
    if not coords_dir.exists():
        raise FileNotFoundError(f"Coordinate directory not found: {coords_dir}")

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, str]] = []
    per_cancer_counts: Dict[str, int] = {}

    for cancer_type in cancer_types:
        src = coords_dir / f"{cancer_type}.tsv"
        if not src.exists():
            raise FileNotFoundError(f"Missing cancer TSV file: {src}")

        fieldnames, rows = _read_tsv_rows(src)
        filtered = [row for row in rows if (row.get("gene") or "").strip() in genes]

        # Keep deterministic order by gene name, then genomic coordinates when present.
        filtered.sort(
            key=lambda r: (
                (r.get("gene") or ""),
                (r.get("chr") or ""),
                int(r.get("start") or 0),
                int(r.get("end") or 0),
            )
        )

        dst = out_dir / f"{cancer_type}.tsv"
        _write_tsv(dst, fieldnames, filtered)

        per_cancer_counts[cancer_type] = len(filtered)

        for row in filtered:
            row_with_cancer = dict(row)
            row_with_cancer["cancer_type"] = cancer_type
            all_rows.append(row_with_cancer)

    merged_path = out_dir / "merged_filtered.tsv"
    merged_fields = ["cancer_type"]
    sample_src = coords_dir / f"{cancer_types[0]}.tsv"
    source_fields, _ = _read_tsv_rows(sample_src)
    merged_fields.extend(source_fields)
    _write_tsv(merged_path, merged_fields, all_rows)

    summary = {
        "cancer_types_file": str(args.cancer_types_file.resolve()),
        "genes_file": str(args.genes_file.resolve()),
        "coords_dir": str(coords_dir),
        "n_cancer_types": len(cancer_types),
        "n_target_genes": len(genes),
        "per_cancer_filtered_rows": per_cancer_counts,
        "n_merged_rows": len(all_rows),
        "output_dir": str(out_dir),
        "output_files": {
            "merged": str(merged_path),
            "per_cancer": [str(out_dir / f"{ct}.tsv") for ct in cancer_types],
        },
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("Filtered driver gene coordinate files created")
    print(f"Cancer types used: {len(cancer_types)}")
    print(f"Target genes used: {len(genes)}")
    print(f"Merged rows written: {len(all_rows)}")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
