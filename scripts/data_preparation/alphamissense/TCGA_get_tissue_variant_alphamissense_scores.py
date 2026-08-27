#!/usr/bin/env python
"""Get AlphaMissense scores for tissue-specific unique SNVs.

Each input TSV must contain ``mutation`` and ``gene`` columns. Mutation values
are expected as ``chrom:start-end:ref:alt``. Before matching, the script removes
MNVs, insertions, deletions, malformed records, and non-ACGT substitutions.

The tissue files are filtered and externally sorted into reusable caches, then
merged against a reusable position-sorted AlphaMissense cache. This keeps
memory usage bounded and lets reruns skip preprocessing. Existing final tissue
TSVs are treated as completed and skipped unless ``--force`` is used.

Example:
    python scripts/data_preparation/get_tissue_variant_alphamissense_scores.py
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, TextIO

from tqdm.auto import tqdm


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_VARIANTS_DIR = PROJECT_DIR / "data" / "alphagenome_scores" / "tissue_unique_variants"
DEFAULT_ALPHAMISSENSE = PROJECT_DIR / "data" / "AlphaMissense_hg38.tsv"
DEFAULT_SORTED_ALPHAMISSENSE = PROJECT_DIR / "data" / "AlphaMissense_hg38.position_sorted.tsv"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data" / "alphagenome_scores" / "tissue_alphamissense_scores"
DEFAULT_PREPROCESSED_DIR = (
    PROJECT_DIR / "data" / "alphagenome_scores" / "tissue_unique_variants_snv_sorted"
)
PREPROCESSED_CACHE_VERSION = 1

DNA_BASES = frozenset("ACGT")
AM_COLUMNS = [
    "genome",
    "uniprot_id",
    "transcript_id",
    "protein_variant",
    "am_pathogenicity",
    "am_class",
]
OUTPUT_COLUMNS = [
    "mutation",
    "gene",
    "CHROM",
    "POS",
    "REF",
    "ALT",
    *AM_COLUMNS,
]

VariantKey = tuple[int, int, str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter tissue variants to SNVs and retrieve AlphaMissense hg38 scores."
    )
    parser.add_argument("--variants-dir", type=Path, default=DEFAULT_VARIANTS_DIR)
    parser.add_argument("--alphamissense", type=Path, default=DEFAULT_ALPHAMISSENSE)
    parser.add_argument(
        "--sorted-alphamissense-cache",
        type=Path,
        default=DEFAULT_SORTED_ALPHAMISSENSE,
        help="Reusable position-sorted AlphaMissense cache.",
    )
    parser.add_argument(
        "--rebuild-alphamissense-cache",
        action="store_true",
        help="Rebuild the position-sorted AlphaMissense cache.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--preprocessed-cache-dir",
        type=Path,
        default=DEFAULT_PREPROCESSED_DIR,
        help="Directory for reusable filtered and sorted tissue SNV caches.",
    )
    parser.add_argument(
        "--rebuild-preprocessed-cache",
        action="store_true",
        help="Rebuild filtered and sorted tissue SNV caches.",
    )
    parser.add_argument(
        "--tissues",
        nargs="+",
        default=None,
        help="Optional tissue names or TSV stems to process (default: all).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess tissues even when completed output TSVs already exist.",
    )
    parser.add_argument(
        "--sort-memory",
        default=None,
        help="Memory limit passed to each concurrent system sort, e.g. 4G.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Parallel tissue filter-and-sort workers (default: %(default)s).",
    )
    parser.add_argument(
        "--max-alphamissense-rows",
        type=int,
        default=None,
        help="Optional AlphaMissense row limit for smoke tests.",
    )
    return parser.parse_args()


def chromosome_rank(chrom: str) -> int | None:
    normalized = chrom.strip()
    if normalized.lower().startswith("chr"):
        normalized = normalized[3:]
    normalized = normalized.upper()
    if normalized.isdigit() and 1 <= int(normalized) <= 22:
        return int(normalized)
    return {"X": 23, "Y": 24, "M": 25, "MT": 25}.get(normalized)


def normalized_chrom(rank: int) -> str:
    if rank <= 22:
        return str(rank)
    return {23: "X", 24: "Y", 25: "MT"}[rank]


def parse_snv(mutation: str) -> tuple[VariantKey, str] | None:
    parts = mutation.strip().split(":")
    if len(parts) != 4:
        return None
    chrom, span, ref, alt = parts
    if "-" not in span:
        return None
    start_text, end_text = span.split("-", 1)
    try:
        start = int(start_text)
        end = int(end_text)
    except ValueError:
        return None

    rank = chromosome_rank(chrom)
    ref = ref.upper()
    alt = alt.upper()
    if (
        rank is None
        or start != end
        or ref not in DNA_BASES
        or alt not in DNA_BASES
        or ref == alt
    ):
        return None
    return (rank, start, ref, alt), f"{normalized_chrom(rank)}:{start}-{end}:{ref}:{alt}"


def parse_alphamissense_key(chrom: str, pos: str, ref: str, alt: str) -> VariantKey | None:
    rank = chromosome_rank(chrom)
    try:
        position = int(pos)
    except ValueError:
        return None
    ref = ref.upper()
    alt = alt.upper()
    if rank is None or ref not in DNA_BASES or alt not in DNA_BASES:
        return None
    return rank, position, ref, alt


class ProgressLines:
    def __init__(self, handle: TextIO, progress: tqdm | None):
        self.handle = handle
        self.progress = progress

    def __iter__(self) -> "ProgressLines":
        return self

    def __next__(self) -> str:
        line = next(self.handle)
        if self.progress is not None:
            self.progress.update(len(line))
        return line


@dataclass
class PreparedTissue:
    name: str
    input_path: Path
    sorted_path: Path
    output_path: Path
    temporary_output_path: Path
    total_rows: int
    snv_rows: int
    filtered_rows: int
    matched_variants: int = 0
    score_rows: int = 0


@dataclass
class TissueCursor:
    tissue: PreparedTissue
    handle: TextIO
    reader: Iterator[list[str]]
    output_handle: TextIO
    writer: csv.writer
    progress: tqdm
    current_key: VariantKey | None = None
    current_rows: list[tuple[str, str]] | None = None
    buffered_row: tuple[VariantKey, str, str] | None = None

    def read_row(self) -> tuple[VariantKey, str, str] | None:
        values = next(self.reader, None)
        if values is None:
            return None
        if len(values) != 6:
            raise ValueError(f"Malformed temporary row for {self.tissue.name}: {values}")
        rank, pos, ref, alt, mutation, gene = values
        return (int(rank), int(pos), ref, alt), mutation, gene

    def advance(self) -> bool:
        first = self.buffered_row or self.read_row()
        self.buffered_row = None
        if first is None:
            self.current_key = None
            self.current_rows = None
            return False

        key, mutation, gene = first
        rows = [(mutation, gene)]
        while True:
            next_row = self.read_row()
            if next_row is None:
                break
            if next_row[0] != key:
                self.buffered_row = next_row
                break
            rows.append((next_row[1], next_row[2]))
        self.current_key = key
        self.current_rows = rows
        return True

    def close(self) -> None:
        self.handle.close()
        self.output_handle.close()


def select_tissue_paths(variants_dir: Path, requested: list[str] | None) -> list[Path]:
    paths = sorted(variants_dir.glob("*.tsv"))
    if requested:
        names = {Path(name).stem for name in requested}
        paths = [path for path in paths if path.stem in names]
        missing = names.difference(path.stem for path in paths)
        if missing:
            raise FileNotFoundError(f"Tissue variant file(s) not found: {sorted(missing)}")
    if not paths:
        raise FileNotFoundError(f"No tissue TSV files found in {variants_dir}")
    return paths


def filter_tissue_file(
    input_path: Path,
    raw_path: Path,
    progress: tqdm | None = None,
) -> tuple[int, int, int]:
    total_rows = 0
    snv_rows = 0
    filtered_rows = 0
    with input_path.open(newline="") as input_handle, raw_path.open("w", newline="") as output_handle:
        header_line = input_handle.readline()
        if progress is not None:
            progress.update(len(header_line))
        if not header_line:
            raise ValueError(f"Empty tissue variant file: {input_path}")
        header = header_line.rstrip("\r\n").split("\t")
        missing = {"mutation", "gene"}.difference(header)
        if missing:
            raise ValueError(f"{input_path} is missing required column(s): {sorted(missing)}")

        mutation_idx = header.index("mutation")
        gene_idx = header.index("gene")
        reader = csv.reader(ProgressLines(input_handle, progress), delimiter="\t")
        writer = csv.writer(output_handle, delimiter="\t", lineterminator="\n")
        for values in reader:
            total_rows += 1
            if mutation_idx >= len(values):
                filtered_rows += 1
                continue
            parsed = parse_snv(values[mutation_idx])
            if parsed is None:
                filtered_rows += 1
                continue
            key, mutation = parsed
            gene = values[gene_idx] if gene_idx < len(values) else ""
            writer.writerow([*key, mutation, gene])
            snv_rows += 1
    return total_rows, snv_rows, filtered_rows


def sort_filtered_file(raw_path: Path, sorted_path: Path, sort_memory: str | None) -> None:
    command = [
        "sort",
        "-t",
        "\t",
        "-k1,1n",
        "-k2,2n",
        "-k3,3",
        "-k4,4",
    ]
    if sort_memory:
        command.extend(["-S", sort_memory])
    command.extend(["-o", str(sorted_path), str(raw_path)])
    environment = os.environ.copy()
    environment["LC_ALL"] = "C"
    subprocess.run(command, check=True, env=environment)


def tissue_cache_paths(cache_dir: Path, input_path: Path) -> tuple[Path, Path]:
    return (
        cache_dir / input_path.name,
        cache_dir / f"{input_path.stem}.metadata.json",
    )


def load_cached_tissue(
    input_path: Path,
    cache_dir: Path,
    output_dir: Path,
) -> PreparedTissue | None:
    sorted_path, metadata_path = tissue_cache_paths(cache_dir, input_path)
    if not sorted_path.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    source_stat = input_path.stat()
    if (
        metadata.get("cache_version") != PREPROCESSED_CACHE_VERSION
        or metadata.get("source_size") != source_stat.st_size
        or metadata.get("source_mtime_ns") != source_stat.st_mtime_ns
    ):
        return None

    try:
        return PreparedTissue(
            name=input_path.stem,
            input_path=input_path,
            sorted_path=sorted_path,
            output_path=output_dir / input_path.name,
            temporary_output_path=output_dir / f".{input_path.name}.tmp",
            total_rows=int(metadata["total_rows"]),
            snv_rows=int(metadata["snv_rows"]),
            filtered_rows=int(metadata["filtered_rows"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def prepare_tissue(
    input_path: Path,
    output_dir: Path,
    cache_dir: Path,
    sort_memory: str | None,
) -> PreparedTissue:
    """Build one persistent tissue cache; designed to run in a worker."""
    sorted_path, metadata_path = tissue_cache_paths(cache_dir, input_path)
    raw_path = cache_dir / f".{input_path.name}.unsorted.tmp"
    temporary_sorted_path = cache_dir / f".{input_path.name}.sorted.tmp"
    temporary_metadata_path = cache_dir / f".{input_path.stem}.metadata.tmp"
    for path in (raw_path, temporary_sorted_path, temporary_metadata_path):
        path.unlink(missing_ok=True)

    try:
        total, snvs, filtered = filter_tissue_file(input_path, raw_path)
        sort_filtered_file(raw_path, temporary_sorted_path, sort_memory)
        source_stat = input_path.stat()
        temporary_metadata_path.write_text(
            json.dumps(
                {
                    "cache_version": PREPROCESSED_CACHE_VERSION,
                    "source_path": str(input_path.resolve()),
                    "source_size": source_stat.st_size,
                    "source_mtime_ns": source_stat.st_mtime_ns,
                    "total_rows": total,
                    "snv_rows": snvs,
                    "filtered_rows": filtered,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        temporary_sorted_path.replace(sorted_path)
        temporary_metadata_path.replace(metadata_path)
    except BaseException:
        temporary_sorted_path.unlink(missing_ok=True)
        temporary_metadata_path.unlink(missing_ok=True)
        raise
    finally:
        raw_path.unlink(missing_ok=True)

    cached = load_cached_tissue(input_path, cache_dir, output_dir)
    if cached is None:
        raise RuntimeError(f"Failed to validate newly built tissue cache: {input_path}")
    return cached


def prepare_tissues(
    tissue_paths: list[Path],
    output_dir: Path,
    cache_dir: Path,
    sort_memory: str | None,
    workers: int,
    force: bool,
) -> list[PreparedTissue]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    prepared_by_index: dict[int, PreparedTissue] = {}
    pending: list[tuple[int, Path]] = []
    if not force:
        for index, input_path in enumerate(tissue_paths):
            cached = load_cached_tissue(input_path, cache_dir, output_dir)
            if cached is None:
                pending.append((index, input_path))
            else:
                prepared_by_index[index] = cached
    else:
        pending = list(enumerate(tissue_paths))

    if prepared_by_index:
        print(f"Reusing preprocessed caches for {len(prepared_by_index)} tissue(s).")
    if not pending:
        return [prepared_by_index[index] for index in range(len(tissue_paths))]

    total_bytes = sum(path.stat().st_size for _, path in pending)
    with (
        tqdm(
            total=len(pending),
            desc="Building tissue caches",
            unit="tissue",
            position=0,
        ) as tissue_progress,
        tqdm(
            total=total_bytes,
            desc="Input processed",
            unit="B",
            unit_scale=True,
            position=1,
        ) as byte_progress,
        ProcessPoolExecutor(max_workers=workers) as executor,
    ):
        futures = {
            executor.submit(
                prepare_tissue,
                input_path,
                output_dir,
                cache_dir,
                sort_memory,
            ): (index, input_path)
            for index, input_path in pending
        }
        for future in as_completed(futures):
            index, input_path = futures[future]
            prepared_by_index[index] = future.result()
            byte_progress.update(input_path.stat().st_size)
            tissue_progress.update(1)
            tissue_progress.set_postfix_str(input_path.stem)

    return [prepared_by_index[index] for index in range(len(tissue_paths))]


def open_cursor(tissue: PreparedTissue, progress: tqdm) -> TissueCursor:
    handle = tissue.sorted_path.open(newline="")
    output_handle = tissue.temporary_output_path.open("w", newline="")
    writer = csv.writer(output_handle, delimiter="\t", lineterminator="\n")
    writer.writerow(OUTPUT_COLUMNS)
    return TissueCursor(
        tissue=tissue,
        handle=handle,
        reader=csv.reader(ProgressLines(handle, progress), delimiter="\t"),
        output_handle=output_handle,
        writer=writer,
        progress=progress,
    )


def read_alphamissense_header(handle: TextIO) -> tuple[list[str], int]:
    bytes_read = 0
    for line in handle:
        bytes_read += len(line)
        if line.startswith("#CHROM\t"):
            return line[1:].rstrip("\r\n").split("\t"), bytes_read
        if not line.startswith("#"):
            return line.rstrip("\r\n").split("\t"), bytes_read
    raise ValueError("AlphaMissense header was not found.")


def build_sorted_alphamissense_cache(
    source_path: Path,
    cache_path: Path,
    sort_memory: str | None,
) -> None:
    """Externally sort AlphaMissense by chromosome rank and position."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_name(f".{cache_path.name}.tmp")
    temporary_path.unlink(missing_ok=True)

    command = ["sort", "-t", "\t", "-k1,1n", "-k2,2n"]
    if sort_memory:
        command.extend(["-S", sort_memory])
    command.extend(["-o", str(temporary_path)])
    environment = os.environ.copy()
    environment["LC_ALL"] = "C"

    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        text=True,
        env=environment,
    )
    try:
        if process.stdin is None:
            raise RuntimeError("Could not open stdin for system sort.")
        with source_path.open(newline="") as source_handle:
            header, bytes_read = read_alphamissense_header(source_handle)
            required = {"CHROM", "POS", "REF", "ALT", *AM_COLUMNS}
            missing = required.difference(header)
            if missing:
                raise ValueError(
                    f"{source_path} is missing required column(s): {sorted(missing)}"
                )
            chrom_idx = header.index("CHROM")
            pos_idx = header.index("POS")

            with tqdm(
                total=source_path.stat().st_size,
                initial=bytes_read,
                desc="Sorting AlphaMissense",
                unit="B",
                unit_scale=True,
            ) as progress:
                for line in source_handle:
                    progress.update(len(line))
                    values = line.rstrip("\r\n").split("\t")
                    if len(values) < len(header):
                        continue
                    rank = chromosome_rank(values[chrom_idx])
                    try:
                        position = int(values[pos_idx])
                    except ValueError:
                        continue
                    if rank is None:
                        continue
                    process.stdin.write(f"{rank}\t{position}\t{line}")
        process.stdin.close()
        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)
        temporary_path.replace(cache_path)
    except BaseException:
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        process.kill()
        process.wait()
        temporary_path.unlink(missing_ok=True)
        raise


def ensure_sorted_alphamissense_cache(
    source_path: Path,
    cache_path: Path,
    sort_memory: str | None,
    force: bool,
) -> Path:
    cache_is_current = (
        cache_path.is_file()
        and cache_path.stat().st_mtime_ns >= source_path.stat().st_mtime_ns
    )
    if cache_is_current and not force:
        print(f"Using sorted AlphaMissense cache: {cache_path}")
        return cache_path

    reason = "forced rebuild" if force else "cache missing or older than source"
    print(f"Building sorted AlphaMissense cache ({reason}): {cache_path}")
    build_sorted_alphamissense_cache(source_path, cache_path, sort_memory)
    return cache_path


def match_scores(
    prepared: list[PreparedTissue],
    alphamissense_path: Path,
    sorted_alphamissense_path: Path,
    max_rows: int | None,
) -> None:
    sorted_progress = tqdm(
        total=sum(tissue.sorted_path.stat().st_size for tissue in prepared),
        desc="Reading sorted SNVs",
        unit="B",
        unit_scale=True,
        position=0,
    )
    cursors: list[TissueCursor] = []
    try:
        cursors = [open_cursor(tissue, sorted_progress) for tissue in prepared]
        heap: list[tuple[VariantKey, int]] = []
        for index, cursor in enumerate(cursors):
            if cursor.advance():
                heapq.heappush(heap, (cursor.current_key, index))

        with alphamissense_path.open(newline="") as source_handle:
            header, _ = read_alphamissense_header(source_handle)
        with sorted_alphamissense_path.open(newline="") as am_handle:
            required = {"CHROM", "POS", "REF", "ALT", *AM_COLUMNS}
            missing = required.difference(header)
            if missing:
                raise ValueError(
                    f"{alphamissense_path} is missing required column(s): {sorted(missing)}"
                )
            column = {name: header.index(name) for name in required}
            column = {name: index + 2 for name, index in column.items()}

            with tqdm(
                total=sorted_alphamissense_path.stat().st_size,
                desc="AlphaMissense",
                unit="B",
                unit_scale=True,
                position=1,
            ) as am_progress:
                current_pos: tuple[int, int] | None = None
                current_rows: dict[tuple[str, str], list[list[str]]] = {}

                def write_matches() -> None:
                    if current_pos is None:
                        return
                    while heap and heap[0][0][:2] < current_pos:
                        _, cursor_index = heapq.heappop(heap)
                        cursor = cursors[cursor_index]
                        if cursor.advance():
                            heapq.heappush(heap, (cursor.current_key, cursor_index))

                    while heap and heap[0][0][:2] == current_pos:
                        key, cursor_index = heapq.heappop(heap)
                        cursor = cursors[cursor_index]
                        am_rows = current_rows.get((key[2], key[3]), [])
                        if am_rows:
                            cursor.tissue.matched_variants += 1
                            for mutation, gene in cursor.current_rows or []:
                                for am_values in am_rows:
                                    cursor.writer.writerow(
                                        [
                                            mutation,
                                            gene,
                                            am_values[column["CHROM"]],
                                            am_values[column["POS"]],
                                            am_values[column["REF"]],
                                            am_values[column["ALT"]],
                                            *[am_values[column[name]] for name in AM_COLUMNS],
                                        ]
                                    )
                                    cursor.tissue.score_rows += 1
                        if cursor.advance():
                            heapq.heappush(heap, (cursor.current_key, cursor_index))

                for row_number, line in enumerate(am_handle, start=1):
                    am_progress.update(len(line))
                    if max_rows is not None and row_number > max_rows:
                        break
                    values = line.rstrip("\r\n").split("\t")
                    if len(values) < len(header):
                        continue
                    key = parse_alphamissense_key(
                        values[column["CHROM"]],
                        values[column["POS"]],
                        values[column["REF"]],
                        values[column["ALT"]],
                    )
                    if key is None:
                        continue
                    pos_key = key[:2]
                    if current_pos is not None and pos_key < current_pos:
                        raise ValueError(
                            f"{sorted_alphamissense_path} is not sorted by genomic position."
                        )
                    if pos_key != current_pos:
                        write_matches()
                        current_pos = pos_key
                        current_rows = {(key[2], key[3]): [values]}
                    else:
                        current_rows.setdefault((key[2], key[3]), []).append(values)
                write_matches()
    except BaseException:
        for cursor in cursors:
            try:
                cursor.close()
            except OSError:
                pass
        sorted_progress.close()
        for tissue in prepared:
            tissue.temporary_output_path.unlink(missing_ok=True)
        raise

    try:
        for cursor in cursors:
            cursor.close()
        sorted_progress.close()
        for tissue in prepared:
            tissue.temporary_output_path.replace(tissue.output_path)
    except BaseException:
        sorted_progress.close()
        for tissue in prepared:
            tissue.temporary_output_path.unlink(missing_ok=True)
        raise


def process(args: argparse.Namespace) -> None:
    selected_paths = select_tissue_paths(args.variants_dir, args.tissues)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for path in selected_paths:
        (args.output_dir / f".{path.name}.tmp").unlink(missing_ok=True)

    completed_paths = [
        path for path in selected_paths
        if (args.output_dir / path.name).is_file()
    ]
    tissue_paths = selected_paths if args.force else [
        path for path in selected_paths
        if path not in completed_paths
    ]

    if completed_paths and not args.force:
        print(f"Skipping {len(completed_paths)} completed tissue(s):")
        for path in completed_paths:
            print(f"  {path.stem}")

    if not tissue_paths:
        print("All requested tissues are already complete. Nothing to do.")
        print(f"Output directory: {args.output_dir}")
        return

    print(f"Processing {len(tissue_paths)} tissue(s).")

    sorted_alphamissense_path = ensure_sorted_alphamissense_cache(
        args.alphamissense,
        args.sorted_alphamissense_cache,
        args.sort_memory,
        args.rebuild_alphamissense_cache,
    )
    prepared = prepare_tissues(
        tissue_paths,
        args.output_dir,
        args.preprocessed_cache_dir,
        args.sort_memory,
        args.workers,
        args.rebuild_preprocessed_cache,
    )
    match_scores(
        prepared,
        args.alphamissense,
        sorted_alphamissense_path,
        args.max_alphamissense_rows,
    )

    print("\nPer-tissue summary:")
    for tissue in prepared:
        print(
            f"  {tissue.name}: {tissue.snv_rows:,} SNV rows, "
            f"{tissue.filtered_rows:,} MNV/indel/malformed rows filtered, "
            f"{tissue.matched_variants:,} variants matched, "
            f"{tissue.score_rows:,} score rows written"
        )
    print(f"Output directory: {args.output_dir}")


def main() -> None:
    args = parse_args()
    if not args.variants_dir.is_dir():
        sys.exit(f"Tissue variants directory not found: {args.variants_dir}")
    if not args.alphamissense.is_file():
        sys.exit(f"AlphaMissense file not found: {args.alphamissense}")
    if shutil.which("sort") is None:
        sys.exit("System 'sort' command was not found.")
    if args.max_alphamissense_rows is not None and args.max_alphamissense_rows < 1:
        sys.exit("--max-alphamissense-rows must be positive.")
    if args.workers < 1:
        sys.exit("--workers must be positive.")

    try:
        process(args)
    except (FileExistsError, FileNotFoundError, ValueError, subprocess.CalledProcessError) as exc:
        sys.exit(f"Error: {exc}")


if __name__ == "__main__":
    main()
