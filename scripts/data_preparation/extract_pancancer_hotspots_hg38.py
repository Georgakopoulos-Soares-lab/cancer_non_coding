#!/usr/bin/env python
"""
Extract retained hotspots in pancancer driver genes and lift them to hg38.

Pancancer driver genes are read from data/driver_genes_coords/Pancancer.tsv.

The output is a TSV with the columns needed for coordinate/gene matching
against files in data/TCGA/tcga_patient_variants:

    chromosome  start  gene

If any hotspot spans more than one base, an end column is also written. The
start/end coordinates are 1-based inclusive hg38 coordinates. Most hotspot
entries are point positions such as chr:pos_count, so end == start and the end
column is omitted.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import os
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path


DRIVER_GENE_FILE = "data/driver_genes_coords/Pancancer.tsv"
HOTSPOT_SNV_FILE = "data/hotspots_v2_snv.csv"
HOTSPOT_INDEL_FILE = "data/hotspots_v2_indel.csv"
HG19_TO_HG38_CHAIN = os.getenv(
    "ONCOGENIE_HG19_TO_HG38_CHAIN",
    "data/ref/hg19ToHg38.over.chain.gz",
)
OUTPUT_FILE = "data/TCGA/pancancer_driver_hotspots_hg38.tsv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract pancancer-driver-gene hotspots and lift coordinates to hg38."
    )
    parser.add_argument(
        "--driver-gene-file",
        type=Path,
        default=DRIVER_GENE_FILE,
        help="Pancancer driver gene coordinate TSV used to define driver genes (default: %(default)s)",
    )
    parser.add_argument("--hotspot-snv-file", type=Path, default=HOTSPOT_SNV_FILE)
    parser.add_argument("--hotspot-indel-file", type=Path, default=HOTSPOT_INDEL_FILE)
    parser.add_argument("--hg19-to-hg38-chain", type=Path, default=HG19_TO_HG38_CHAIN)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    return parser.parse_args()


def normalize_chromosome(chrom: object) -> str | None:
    if chrom is None:
        return None
    chrom_s = str(chrom).strip()
    if not chrom_s or chrom_s.lower() == "nan":
        return None
    if chrom_s.lower().startswith("chr"):
        chrom_s = chrom_s[3:]
    return chrom_s


def read_pancancer_genes(path: Path) -> set[str]:
    genes: set[str] = set()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or "gene" not in reader.fieldnames:
            raise ValueError(f"{path} must contain a 'gene' column")
        for row in reader:
            gene = row.get("gene", "").strip()
            if gene:
                genes.add(gene)
    return genes


def open_text_maybe_gzip(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return path.open("r")


class ChainLiftOver:
    """Minimal UCSC chain liftover for point/short interval coordinates."""

    def __init__(self, chain_path: Path):
        if not chain_path.exists():
            raise FileNotFoundError(f"LiftOver chain not found: {chain_path}")
        self.blocks_by_chrom: dict[str, list[tuple[int, int, int, str, int]]] = defaultdict(list)
        self.starts_by_chrom: dict[str, list[int]] = {}
        self._load(chain_path)

    def _load(self, chain_path: Path) -> None:
        current = None
        with open_text_maybe_gzip(chain_path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    current = None
                    continue
                if line.startswith("chain "):
                    parts = line.split()
                    current = {
                        "src_chrom": normalize_chromosome(parts[2]),
                        "src_pos": int(parts[5]),
                        "dst_size": int(parts[8]),
                        "dst_strand": parts[9],
                        "dst_pos": int(parts[10]),
                    }
                    continue
                if current is None:
                    continue

                vals = [int(x) for x in line.split()]
                size = vals[0]
                src_start = current["src_pos"]
                dst_start = current["dst_pos"]
                self.blocks_by_chrom[current["src_chrom"]].append(
                    (src_start, src_start + size, dst_start, current["dst_strand"], current["dst_size"])
                )
                current["src_pos"] += size
                current["dst_pos"] += size
                if len(vals) == 3:
                    current["src_pos"] += vals[1]
                    current["dst_pos"] += vals[2]

        for chrom, blocks in self.blocks_by_chrom.items():
            blocks.sort(key=lambda x: x[0])
            self.starts_by_chrom[chrom] = [b[0] for b in blocks]

    def lift_point0(self, chrom: str, pos0: int) -> tuple[str, int] | None:
        chrom = normalize_chromosome(chrom)
        if chrom is None:
            return None
        starts = self.starts_by_chrom.get(chrom, [])
        blocks = self.blocks_by_chrom.get(chrom, [])
        idx = bisect_right(starts, pos0) - 1
        if idx < 0:
            return None
        src_start, src_end, dst_start, dst_strand, dst_size = blocks[idx]
        if not (src_start <= pos0 < src_end):
            return None
        offset = pos0 - src_start
        if dst_strand == "-":
            dst0 = dst_size - (dst_start + offset) - 1
        else:
            dst0 = dst_start + offset
        return chrom, dst0

    def lift_interval1(self, chrom: str, start1: int, end1: int) -> tuple[str, int, int] | None:
        lifted_start = self.lift_point0(chrom, start1 - 1)
        lifted_end = self.lift_point0(chrom, end1 - 1)
        if lifted_start is None or lifted_end is None:
            return None
        chrom_a, start0 = lifted_start
        chrom_b, end0 = lifted_end
        if chrom_a != chrom_b:
            return None
        return chrom_a, min(start0, end0) + 1, max(start0, end0) + 1


def parse_genomic_position(value: str) -> list[tuple[str, int, int, str]]:
    intervals = []
    for token in str(value).split("|"):
        token = token.strip()
        if not token or ":" not in token:
            continue
        chrom, coord = token.split(":", 1)
        coord_no_count = coord.split("_", 1)[0]
        try:
            if "-" in coord_no_count:
                raw_start, raw_end = map(int, coord_no_count.split("-", 1))
                start1, end1 = min(raw_start, raw_end), max(raw_start, raw_end)
            else:
                start1 = end1 = int(coord_no_count)
        except ValueError:
            continue
        chrom = normalize_chromosome(chrom)
        if chrom is not None:
            intervals.append((chrom, start1, end1, token))
    return intervals


def hotspot_rows(path: Path, source: str, genes: set[str]):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"Hugo_Symbol", "Genomic_Position"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"{path} must contain columns: {sorted(required)}")
        for row in reader:
            gene = row.get("Hugo_Symbol", "").strip()
            if gene not in genes:
                continue
            for chrom, start1, end1, raw_position in parse_genomic_position(row.get("Genomic_Position", "")):
                yield source, row, chrom, start1, end1, raw_position


def main() -> None:
    args = parse_args()

    pancancer_genes = read_pancancer_genes(args.driver_gene_file)
    print(
        f"Pancancer driver genes from {args.driver_gene_file}: "
        f"{len(pancancer_genes)} unique gene(s)"
    )
    lifter = ChainLiftOver(args.hg19_to_hg38_chain)
    rows = []
    n_raw = 0
    n_lifted = 0
    seen_hotspots = set()

    for hotspot_file, source in [(args.hotspot_snv_file, "snv"), (args.hotspot_indel_file, "indel")]:
        for source, row, chrom, start1, end1, raw_position in hotspot_rows(
            hotspot_file, source, pancancer_genes
        ):
            n_raw += 1
            lifted = lifter.lift_interval1(chrom, start1, end1)
            if lifted is None:
                continue
            hg38_chrom, hg38_start1, hg38_end1 = lifted
            n_lifted += 1
            key = (hg38_chrom, hg38_start1, hg38_end1, row["Hugo_Symbol"])
            if key in seen_hotspots:
                continue
            seen_hotspots.add(key)
            rows.append(
                {
                    "chromosome": hg38_chrom,
                    "start": hg38_start1,
                    "end": hg38_end1,
                    "gene": row["Hugo_Symbol"],
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    has_ranges = any(row["start"] != row["end"] for row in rows)
    fieldnames = ["chromosome", "start", "gene"] if not has_ranges else ["chromosome", "start", "end", "gene"]
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})

    output_genes = {row["gene"] for row in rows}
    print(f"Hotspot intervals considered: {n_raw}")
    print(f"Hotspot intervals lifted to hg38: {n_lifted}")
    print(f"Output rows: {len(rows)}")
    print(f"Output unique genes: {len(output_genes)}")
    print(f"Output has ranged hotspots: {has_ranges}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
