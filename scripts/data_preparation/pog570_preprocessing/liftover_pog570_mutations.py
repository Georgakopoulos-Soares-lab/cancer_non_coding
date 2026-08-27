#!/usr/bin/env python
"""Lift over all PCAWG and POG570 per-patient mutation files from hg19 to hg38.

Input files are read from data/PCAWG_POG570_mutations. Output files are written
to data/PCAWG_POG570_mutations_hg38 with the same file names. The output keeps
the original mutation in hg19_mutation and replaces mutation with the lifted
hg38 coordinate. Rows that cannot be lifted are omitted.
"""

import gzip
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm


PROJECT_DIR = Path(__file__).resolve().parents[3]
INPUT_DIR = PROJECT_DIR / "data" / "PCAWG_POG570_mutations"
OUTPUT_DIR = PROJECT_DIR / "data" / "PCAWG_POG570_mutations_hg38"
CHAIN_PATH = PROJECT_DIR / "data" / "ref" / "hg19ToHg38.over.chain.gz"
N_JOBS = 8
WORKER_LIFTER = None


def normalise_chrom(chrom: str) -> str:
    chrom = str(chrom)
    return chrom[3:] if chrom.lower().startswith("chr") else chrom


def normalise_mutation(mutation: str) -> str:
    parts = str(mutation).split(":")
    if len(parts) != 4:
        return str(mutation)
    chrom, coord, ref, alt = parts
    if "-" not in coord:
        coord = f"{int(coord)}-{int(coord)}"
    return f"{chrom}:{coord}:{ref}:{alt}"


def parse_mutation_interval(mutation: str):
    chrom, coords, ref, alt = normalise_mutation(mutation).split(":")
    start, end = map(int, coords.split("-"))
    return normalise_chrom(chrom), min(start, end) - 1, max(start, end), ref.replace("-", ""), alt.replace("-", "")


class ChainLifter:
    def __init__(self, chain_path: Path):
        self.blocks_by_chrom: dict[str, list[tuple[int, int, str, int]]] = {}
        self._load(chain_path)

    def _load(self, chain_path: Path) -> None:
        opener = gzip.open if str(chain_path).endswith(".gz") else open
        with opener(chain_path, "rt") as handle:
            target_chrom = None
            target_pos = query_pos = 0
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith("chain "):
                    fields = line.split()
                    target_chrom = normalise_chrom(fields[2])
                    target_pos = int(fields[5])
                    query_chrom = fields[7]
                    query_size = int(fields[8])
                    query_strand = fields[9]
                    query_pos = int(fields[10]) if query_strand == "+" else query_size - int(fields[11])
                    continue

                parts = [int(value) for value in line.split()]
                size = parts[0]
                self.blocks_by_chrom.setdefault(target_chrom, []).append(
                    (target_pos, target_pos + size, query_chrom, query_pos)
                )
                target_pos += size
                query_pos += size
                if len(parts) == 3:
                    target_pos += parts[1]
                    query_pos += parts[2]

        for blocks in self.blocks_by_chrom.values():
            blocks.sort()

    def lift_point(self, chrom: str, pos0: int) -> tuple[str, int] | None:
        chrom = normalise_chrom(chrom)
        for target_start, target_end, query_chrom, query_start in self.blocks_by_chrom.get(chrom, []):
            if target_start <= pos0 < target_end:
                return query_chrom, query_start + (pos0 - target_start)
        return None


def liftover_mutation(mutation: str, lifter: ChainLifter) -> str | None:
    chrom, start0, end0, ref, alt = parse_mutation_interval(mutation)
    lifted_start = lifter.lift_point(chrom, start0)
    lifted_end = lifter.lift_point(chrom, max(start0, end0 - 1))
    if lifted_start is None or lifted_end is None or lifted_start[0] != lifted_end[0]:
        return None

    out_start = lifted_start[1] + 1
    out_end = lifted_end[1] + 1
    coord = str(out_start) if out_start == out_end else f"{out_start}-{out_end}"
    return normalise_mutation(f"{lifted_start[0]}:{coord}:{ref or '-'}:{alt or '-'}")


def liftover_file(path: Path, output_path: Path, lifter: ChainLifter) -> tuple[str, int, int, int]:
    df = pd.read_csv(path, sep="\t", dtype=str)
    n_input = len(df)
    df = df.dropna(subset=["mutation"]).copy()
    n_with_mutation = len(df)
    df["hg19_mutation"] = df["mutation"].map(normalise_mutation)
    df["mutation"] = df["hg19_mutation"].map(lambda value: liftover_mutation(value, lifter))
    df = df.dropna(subset=["mutation"])
    n_lifted = len(df)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep="\t", index=False)
    return path.name, n_input, n_with_mutation, n_lifted


def init_worker(chain_path: Path) -> None:
    global WORKER_LIFTER
    WORKER_LIFTER = ChainLifter(chain_path)


def liftover_file_task(path: Path) -> tuple[str, int, int, int]:
    return liftover_file(path, OUTPUT_DIR / path.name, WORKER_LIFTER)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lift over PCAWG/POG570 mutation files from hg19 to hg38.")
    parser.add_argument("--n-jobs", type=int, default=N_JOBS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(INPUT_DIR.glob("*.tsv"))
    results = []
    if args.n_jobs == 1:
        lifter = ChainLifter(CHAIN_PATH)
        for path in tqdm(paths, desc="PCAWG/POG570 liftover", unit="patient"):
            results.append(liftover_file(path, OUTPUT_DIR / path.name, lifter))
    else:
        with ProcessPoolExecutor(max_workers=args.n_jobs, initializer=init_worker, initargs=(CHAIN_PATH,)) as executor:
            futures = [executor.submit(liftover_file_task, path) for path in paths]
            for future in tqdm(as_completed(futures), total=len(futures), desc="PCAWG/POG570 liftover", unit="patient"):
                results.append(future.result())

    total_input = sum(result[1] for result in results)
    total_with_mutation = sum(result[2] for result in results)
    total_lifted = sum(result[3] for result in results)
    total_missing_mutation = total_input - total_with_mutation
    total_dropped_liftover = total_with_mutation - total_lifted
    print(
        "Liftover summary: "
        f"files={len(results)}, input_rows={total_input:,}, "
        f"missing_mutation={total_missing_mutation:,}, "
        f"attempted_liftover={total_with_mutation:,}, "
        f"lifted={total_lifted:,}, "
        f"dropped_liftover={total_dropped_liftover:,}"
    )
    for name, n_input, n_with_mutation, n_lifted in sorted(results):
        dropped_liftover = n_with_mutation - n_lifted
        if dropped_liftover:
            print(
                f"{name}: input_rows={n_input:,}, attempted_liftover={n_with_mutation:,}, "
                f"lifted={n_lifted:,}, dropped_liftover={dropped_liftover:,}"
            )
    print(f"Wrote lifted mutation files to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
