#!/usr/bin/env bash

set -euo pipefail

GDC_CLIENT="scripts/1.1_download_TCGA_data/gdc-client"
CHUNK_SIZE=10
MANIFEST_FILE="$1"
OUTPUT_DIR="$2"

if [[ -z "$MANIFEST_FILE" ]]; then
	echo "Usage: $0 <manifest_file> <output_dir>"
	exit 1
fi
if [[ -z "$OUTPUT_DIR" ]]; then
	echo "Usage: $0 <manifest_file> <output_dir>"
	exit 1
fi

# # Gene Expression data
# MANIFEST_FILE="scripts/1.1_download_TCGA_data/gene_exp/gdc_manifest.2026-02-20.191943.txt"
# OUTPUT_DIR="data/TCGA/Raw_Data/gene_expression"
# # CNV data
# MANIFEST_FILE="scripts/1.1_download_TCGA_data/cnv/gdc_manifest.2026-02-20.211903.txt"
# OUTPUT_DIR="data/TCGA/Raw_Data/cnv"

mkdir -p "$OUTPUT_DIR"

header="$(head -n 1 "$MANIFEST_FILE")"

chunk_file="$(mktemp)"
cleanup() {
	rm -f "$chunk_file"
}
trap cleanup EXIT

start_new_chunk() {
	: > "$chunk_file"
	printf '%s\n' "$header" > "$chunk_file"
}

download_chunk() {
	local count="$1"
	if (( count > 0 )); then
		"./$GDC_CLIENT" download -m "$chunk_file" -d "$OUTPUT_DIR"
		rm -f "$chunk_file"
		chunk_file="$(mktemp)"
	fi
}

files_in_chunk=0
start_new_chunk
while IFS= read -r line; do
	[[ -z "$line" ]] && continue
	printf '%s\n' "$line" >> "$chunk_file"
	files_in_chunk=$((files_in_chunk + 1))
	if (( files_in_chunk == CHUNK_SIZE )); then
		download_chunk "$files_in_chunk"
		files_in_chunk=0
		start_new_chunk
	fi
done < <(tail -n +2 "$MANIFEST_FILE")
download_chunk "$files_in_chunk"
