#!/usr/bin/env python3
"""Extract tumor VAFs from every VCF below an input directory."""

import argparse
import csv
import gzip
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock


OUTPUT_COLUMNS = [
    "input_vcf",
    "tumor_sample_id",
    "chrom",
    "position",
    "ref",
    "alt",
    "filter",
    "vaf",
    "vaf_source",
]


def find_vcfs(vcf_dir):
    return sorted(path for path in Path(vcf_dir).rglob("*.vcf*") if path.is_file())


def tumor_column(header):
    samples = header[9:]
    for index, sample_id in enumerate(samples, start=9):
        if sample_id.upper() in {"TUMOR", "TUMOUR"}:
            return index, sample_id
    for index, sample_id in enumerate(samples, start=9):
        parts = sample_id.split("-")
        if len(parts) >= 4 and parts[3][:2].isdigit() and int(parts[3][:2]) < 10:
            return index, sample_id
    return 9, samples[0] if samples else ""


def value_at(values, index):
    return values[index] if index < len(values) and values[index] not in {"", "."} else ""


def vaf(format_keys, sample, allele_index):
    values = dict(zip(format_keys.split(":"), sample.split(":")))
    for key in ("AF", "FREQ"):
        value = value_at(values.get(key, "").replace("%", "").split(","), allele_index)
        if value:
            fraction = float(value)
            return (fraction / 100 if key == "FREQ" or fraction > 1 else fraction), key

    ad = values.get("AD", "").split(",")
    alt_depth = value_at(ad, allele_index + 1)
    total_depth = sum(int(value) for value in ad if value not in {"", "."})
    if alt_depth and total_depth:
        return int(alt_depth) / total_depth, "AD"

    alt_depth = value_at(values.get("AO", "").split(","), allele_index)
    ref_depth = values.get("RO", "")
    if alt_depth and ref_depth:
        return int(alt_depth) / (int(alt_depth) + int(ref_depth)), "AO/RO"
    return "", ""


def extract_vafs(vcf_path, output_dir, pass_only):
    output_path = output_dir / f"{vcf_path.name}.vaf.tsv.gz"
    if output_path.exists():
        return "skipped"

    partial_path = output_path.with_name(f"{output_path.name}.part")
    opener = gzip.open if vcf_path.suffix == ".gz" else open
    with opener(vcf_path, "rt") as vcf, gzip.open(partial_path, "wt", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=OUTPUT_COLUMNS, delimiter="\t")
        writer.writeheader()
        sample_index, sample_id = 9, ""
        for line in vcf:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                sample_index, sample_id = tumor_column(line.rstrip().split("\t"))
                continue
            fields = line.rstrip().split("\t")
            if len(fields) <= sample_index or (pass_only and fields[6] != "PASS"):
                continue
            for allele_index, alt in enumerate(fields[4].split(",")):
                fraction, source = vaf(fields[8], fields[sample_index], allele_index)
                if fraction == "":
                    continue
                writer.writerow(
                    {
                        "input_vcf": str(vcf_path),
                        "tumor_sample_id": sample_id,
                        "chrom": fields[0],
                        "position": fields[1],
                        "ref": fields[3],
                        "alt": alt,
                        "filter": fields[6],
                        "vaf": f"{fraction:.8g}",
                        "vaf_source": source,
                    }
                )
    partial_path.replace(output_path)
    return "processed"


class Progress:
    def __init__(self, total):
        self.total = total
        self.started = 0
        self.completed = 0
        self.lock = Lock()

    def render(self):
        width = 30
        filled = int(width * self.completed / self.total)
        bar = "#" * filled + "-" * (width - filled)
        print(f"\r[{bar}] started {self.started}/{self.total}, completed {self.completed}/{self.total}", end="", flush=True)

    def start(self):
        with self.lock:
            self.started += 1
            self.render()

    def complete(self):
        with self.lock:
            self.completed += 1
            self.render()


def process_vcf(vcf_path, output_dir, pass_only, progress):
    progress.start()
    return extract_vafs(vcf_path, output_dir, pass_only)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vcf-dir",
        default="/home1/10900/aksh/group/group_resources/TCGA_Dataset/TCGA_VCFs/snv_indels",
        help="Directory containing VCF files.",
    )
    parser.add_argument("--output-dir", default="data/TCGA/VAFs", help="Directory to write VAF TSVs.")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent VCF extractions.")
    parser.add_argument("--pass-only", action="store_true", help="Write only VCF records with FILTER=PASS.")
    args = parser.parse_args()

    vcfs = find_vcfs(args.vcf_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(vcfs)} VCFs in {args.vcf_dir}.")
    if not vcfs:
        return

    progress = Progress(len(vcfs))
    progress.render()
    counts = {"processed": 0, "skipped": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_vcf, path, output_dir, args.pass_only, progress) for path in vcfs]
        for future in as_completed(futures):
            counts[future.result()] += 1
            progress.complete()
    print(f"\nFinished: {counts['processed']} processed, {counts['skipped']} already present.")


if __name__ == "__main__":
    main()
