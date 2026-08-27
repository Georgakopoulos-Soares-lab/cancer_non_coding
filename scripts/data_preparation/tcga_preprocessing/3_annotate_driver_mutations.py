#!/usr/bin/env python
"""Annotate TCGA driver-gene/hotspot variants with PCAWG driver mutations.

PCAWG driver mutations are supplied in hg19 coordinates.  They are lifted to
hg38 before being matched to the hg38 TCGA variants produced by
2_annotate_hotspots_driver_genes.py.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_DIR = PROJECT_DIR / "data" / "TCGA" / "tcga_driver_gene_hotspot_variants_by_cancer"
DEFAULT_DRIVER_MUTATIONS = (
    PROJECT_DIR
    / "data"
    / "PCAWG_driver_mutations"
    / "TableS3_panorama_driver_mutations_TCGA_samples.controlled.tsv"
)
DEFAULT_CHAIN = PROJECT_DIR / "data" / "ref" / "hg19ToHg38.over.chain.gz"
DEFAULT_CANCER_ORGAN_MAP = PROJECT_DIR / "metadata" / "cancer_types_acronyms.tsv"
DEFAULT_PCAWG_TISSUE_MAP = PROJECT_DIR / "metadata" / "cancer_tissue_cell_lines.tsv"
DEFAULT_LIFTED_DRIVER_MUTATIONS = (
    PROJECT_DIR / "data" / "PCAWG_driver_mutations" / "driver_mutations_hg19_lifted_to_hg38.tsv"
)
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "data" / "TCGA" / "tcga_driver_gene_hotspot_driver_variants_by_cancer"
CHUNK_SIZE = 500_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--driver-mutations", type=Path, default=DEFAULT_DRIVER_MUTATIONS)
    parser.add_argument("--hg19-to-hg38-chain", type=Path, default=DEFAULT_CHAIN)
    parser.add_argument("--cancer-organ-map", type=Path, default=DEFAULT_CANCER_ORGAN_MAP)
    parser.add_argument("--pcawg-tissue-map", type=Path, default=DEFAULT_PCAWG_TISSUE_MAP)
    parser.add_argument("--lifted-driver-mutations", type=Path, default=DEFAULT_LIFTED_DRIVER_MUTATIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--overwrite", action="store_true", help="Replace files already in the new output directory.")
    return parser.parse_args()


def load_liftover(chain_path: Path):
    helper_path = PROJECT_DIR / "scripts" / "data_preparation" / "6_extract_pancancer_hotspots_hg38.py"
    spec = importlib.util.spec_from_file_location("liftover_helpers", helper_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load liftover helper from {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ChainLiftOver(chain_path)


def normalize_chrom(chrom: object) -> str:
    chrom = str(chrom).strip()
    return chrom[3:] if chrom.lower().startswith("chr") else chrom


def mutation_key(mutation: object, gene: object) -> tuple[str, int, int, str] | None:
    parts = str(mutation).split(":")
    if len(parts) != 4:
        return None
    chrom, coords, _, _ = parts
    try:
        start, end = map(int, coords.split("-", 1)) if "-" in coords else (int(coords), int(coords))
    except ValueError:
        return None
    return normalize_chrom(chrom), min(start, end), max(start, end), str(gene)


def driver_interval(row: pd.Series, lifter) -> tuple[str, int, int, str] | None:
    chrom = normalize_chrom(row["chr"])
    try:
        start = int(row["pos"])
    except ValueError:
        return None
    ref = str(row["ref"])
    end = start + len(ref) - 1 if ref not in {"", "-"} else start
    lifted = lifter.lift_interval1(chrom, start, end)
    if lifted is None:
        return None
    lifted_chrom, lifted_start, lifted_end = lifted
    return lifted_chrom, lifted_start, lifted_end, str(row["gene"])


def liftover_driver_mutations(path: Path, output_path: Path, lifter) -> pd.DataFrame:
    drivers = pd.read_csv(path, sep="\t", dtype=str, usecols=["ttype", "chr", "pos", "ref", "alt", "gene"])
    drivers = drivers[(drivers["chr"] != "x") & (drivers["pos"] != "x")].dropna()
    drivers = drivers.rename(columns={"chr": "hg19_chromosome", "pos": "hg19_start"})
    drivers["hg19_chromosome"] = drivers["hg19_chromosome"].map(normalize_chrom)
    drivers["hg19_start"] = drivers["hg19_start"].astype(int)
    drivers["hg19_end"] = drivers.apply(
        lambda row: row["hg19_start"] + len(row["ref"]) - 1 if row["ref"] != "-" else row["hg19_start"],
        axis=1,
    )
    lifted = [
        driver_interval(row.rename({"hg19_chromosome": "chr", "hg19_start": "pos"}), lifter)
        for _, row in drivers.iterrows()
    ]
    drivers["lifted"] = lifted
    drivers = drivers[drivers["lifted"].notna()].copy()
    drivers[["hg38_chromosome", "hg38_start", "hg38_end", "lifted_gene"]] = pd.DataFrame(
        drivers["lifted"].tolist(), index=drivers.index
    )
    drivers = drivers.drop(columns=["lifted", "lifted_gene"]).drop_duplicates()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    drivers.to_csv(output_path, sep="\t", index=False)
    return drivers


def add_ttype_to_lifted_mutations(lifted_path: Path, driver_path: Path) -> None:
    lifted = pd.read_csv(lifted_path, sep="\t", dtype=str)
    raw = pd.read_csv(driver_path, sep="\t", dtype=str, usecols=["ttype", "chr", "pos", "ref", "gene"])
    raw = raw[(raw["chr"] != "x") & (raw["pos"] != "x")].dropna()
    raw = raw.rename(columns={"chr": "hg19_chromosome", "pos": "hg19_start"})
    raw["hg19_chromosome"] = raw["hg19_chromosome"].map(normalize_chrom)
    merge_columns = ["hg19_chromosome", "hg19_start", "ref", "gene"]
    lifted = lifted.drop(columns=["ttype"], errors="ignore").merge(
        raw.drop_duplicates(), on=merge_columns, how="left"
    )
    lifted.to_csv(lifted_path, sep="\t", index=False)


def load_cancer_hotspot_organs(cancer_map_path: Path) -> dict[str, set[str]]:
    mapping = pd.read_csv(cancer_map_path, sep="\t", dtype=str).fillna("")
    return {
        acronym: {organ.strip() for organ in organs.split(",") if organ.strip()}
        for acronym, organs in mapping[mapping["Projects"].str.contains("TCGA")].groupby("Acronym")["Hotspot cohort"].agg(",".join).items()
    }


def load_pcawg_hotspot_organs(cancer_map_path: Path, tissue_map_path: Path) -> dict[str, set[str]]:
    cancer_map = pd.read_csv(cancer_map_path, sep="\t", dtype=str).fillna("")
    tissue_map = pd.read_csv(tissue_map_path, sep="\t", dtype=str)
    tissue_aliases = {"Bone": "Bone_SoftTissue", "Colon": "Colon_Rectum", "Head": "Head_Neck"}
    tissue_map["organ"] = tissue_map["tissue"].replace(tissue_aliases)
    tissue_map = tissue_map.rename(columns={"cancer_type": "ttype"})
    organs = cancer_map[["Organ", "Hotspot cohort"]].rename(columns={"Organ": "organ"})
    mapped = tissue_map.merge(organs, on="organ", how="left")
    result = {
        ttype: {organ.strip() for organ in cohorts.dropna() for organ in organ.split(",") if organ.strip()}
        for ttype, cohorts in mapped.groupby("ttype")["Hotspot cohort"]
    }
    result.update({"Cervix-AdenoCA": {"cervix"}, "Cervix-SCC": {"cervix"}})
    return result


def load_driver_keys_by_organ(
    path: Path, pcawg_hotspot_organs: dict[str, set[str]]
) -> dict[str, set[tuple[str, int, int, str]]]:
    drivers = pd.read_csv(
        path,
        sep="\t",
        dtype={"ttype": str, "hg38_chromosome": str, "hg38_start": int, "hg38_end": int, "gene": str},
        usecols=["ttype", "hg38_chromosome", "hg38_start", "hg38_end", "gene"],
    )
    keys_by_organ: dict[str, set[tuple[str, int, int, str]]] = {}
    for row in drivers.itertuples(index=False):
        key = (row.hg38_chromosome, row.hg38_start, row.hg38_end, row.gene)
        for organ in pcawg_hotspot_organs.get(row.ttype, set()):
            keys_by_organ.setdefault(organ, set()).add(key)
    return keys_by_organ


def annotate_file(
    input_path: Path,
    output_path: Path,
    driver_keys: set[tuple[str, int, int, str]],
    chunk_size: int,
    overwrite: bool,
) -> tuple[int, int]:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite to replace it.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_variants = n_drivers = 0
    wrote_header = False
    for chunk in pd.read_csv(input_path, sep="\t", dtype=str, chunksize=chunk_size):
        keys = [mutation_key(mutation, gene) for mutation, gene in zip(chunk["mutation"], chunk["gene_name"])]
        chunk["is_driver"] = [key in driver_keys if key is not None else False for key in keys]
        chunk["is_driver_hotspot"] = chunk["is_driver"] | chunk["is_hotspot"].eq("True")
        chunk.to_csv(output_path, sep="\t", index=False, mode="w" if not wrote_header else "a", header=not wrote_header)
        wrote_header = True
        n_variants += len(chunk)
        n_drivers += int(chunk["is_driver"].sum())
    return n_variants, n_drivers


def main() -> None:
    args = parse_args()
    if args.lifted_driver_mutations.exists():
        print(f"Using existing lifted driver mutations: {args.lifted_driver_mutations}")
    else:
        lifter = load_liftover(args.hg19_to_hg38_chain)
        lifted = liftover_driver_mutations(args.driver_mutations, args.lifted_driver_mutations, lifter)
        print(f"Lifted {len(lifted):,} PCAWG driver mutations to {args.lifted_driver_mutations}")
    lifted_columns = pd.read_csv(args.lifted_driver_mutations, sep="\t", nrows=0).columns
    if "ttype" not in lifted_columns:
        add_ttype_to_lifted_mutations(args.lifted_driver_mutations, args.driver_mutations)
        print(f"Added PCAWG tumour types to {args.lifted_driver_mutations}")
    cancer_hotspot_organs = load_cancer_hotspot_organs(args.cancer_organ_map)
    pcawg_hotspot_organs = load_pcawg_hotspot_organs(args.cancer_organ_map, args.pcawg_tissue_map)
    driver_keys_by_organ = load_driver_keys_by_organ(args.lifted_driver_mutations, pcawg_hotspot_organs)
    print(f"Loaded driver-mutation keys for {len(driver_keys_by_organ)} hotspot organ(s).")

    results = []
    for input_path in sorted(args.input_dir.glob("*_variants.tsv")):
        cancer_type = input_path.name.removesuffix("_variants.tsv")
        driver_keys = set().union(*(driver_keys_by_organ.get(organ, set()) for organ in cancer_hotspot_organs.get(cancer_type, set())))
        output_path = args.output_dir / input_path.name
        n_variants, n_drivers = annotate_file(
            input_path, output_path, driver_keys, args.chunk_size, args.overwrite
        )
        results.append((input_path.name, n_variants, n_drivers))

    for name, n_variants, n_drivers in results:
        print(f"{name}: variants={n_variants:,}, driver_mutations={n_drivers:,}")
    print(f"Wrote {len(results)} file(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
