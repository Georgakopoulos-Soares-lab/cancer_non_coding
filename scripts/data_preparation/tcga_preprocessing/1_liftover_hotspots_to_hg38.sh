#!/usr/bin/env bash
set -euo pipefail

# Lift TCGA hotspots from hg19 to hg38.

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

INPUT_BED="data/hotspots_v2/hotspots_tcga_hg19.bed"
OUTPUT_BED="data/hotspots_v2/hotspots_tcga_hg38.bed"
UNMAPPED_BED="data/hotspots_v2/hotspots_tcga_hg38_unmapped.bed"
CHAIN_FILE="metadata/hg19ToHg38.over.chain.gz"
LIFTOVER_BIN="scripts/liftOver"

if [[ ! -f "$INPUT_BED" ]]; then
  echo "Input file not found: $INPUT_BED" >&2
  exit 1
fi

if [[ ! -f "$CHAIN_FILE" ]]; then
  echo "Chain file not found: $CHAIN_FILE" >&2
  exit 1
fi

if [[ ! -x "$LIFTOVER_BIN" ]]; then
  echo "liftOver binary not found or not executable: $LIFTOVER_BIN" >&2
  echo "Download UCSC liftOver and place it at scripts/liftOver" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_BED")"

"$LIFTOVER_BIN" "$INPUT_BED" "$CHAIN_FILE" "$OUTPUT_BED" "$UNMAPPED_BED"

echo "LiftOver complete."
echo "Mapped:   $OUTPUT_BED"
echo "Unmapped: $UNMAPPED_BED"
