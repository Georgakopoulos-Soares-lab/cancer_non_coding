#!/usr/bin/env bash

# This script renames files downloaded from GDC to have the subdirectory name as a prefix.
# For example, it renames: gene_expression/12345/file.tsv -> gene_expression/12345.tsv
# This is needed because the GDC client downloads files into subdirectories named after the patient ID, and we want to flatten this structure for easier access.

set -euo pipefail

INPUT_DIR="${1:-}" # e.g. "data/TCGA/Raw_Data/gene_expression" or "data/TCGA/Raw_Data/cnv"
OUTPUT_DIR="${2:-}" # e.g. "data/TCGA/gene_expression_data" or "data/TCGA/cnv_data"
FILE_EXT="${3:-}" # e.g. "tsv" (for gene_expression) or "seg.txt" (for cnv)

mkdir -p "$OUTPUT_DIR"

if [[ -z "${INPUT_DIR}" || -z "${OUTPUT_DIR}" || -z "${FILE_EXT}" ]]; then
	echo "Usage: $0 <input_dir> <output_dir> <file_ext>"
	echo "Example: $0 data/TCGA/Raw_Data/gene_expression data/TCGA/gene_expression_data tsv"
	exit 1
fi

if [[ ! -d "$INPUT_DIR" ]]; then
	echo "Directory not found: $INPUT_DIR"
	exit 1
fi

# count all files directly within a subdirectory of the output directory
echo "Number of data files found:"
find "$INPUT_DIR" -mindepth 2 -maxdepth 2 -type f \( -name "*.${FILE_EXT}" \) | wc -l

# echo "Subdirectories with multiple direct .${FILE_EXT} files:"
# find "$INPUT_DIR" -mindepth 1 -maxdepth 1 -type d -print0 | while IFS= read -r -d '' subdir; do
# 	count=$(find "$subdir" -mindepth 1 -maxdepth 1 -type f -name "*.${FILE_EXT}" | wc -l)
# 	if (( count > 1 )); then
# 		echo "$subdir ($count)"
# 	fi
# done

# rename files to have subdirectory name as prefix, 
# e.g. data/TCGA/gene_expression/12345/file.tsv -> data/TCGA/gene_expression/12345.tsv
find "$INPUT_DIR" -mindepth 2 -maxdepth 2 -type f \( -name "*.${FILE_EXT}" \) | while read -r file; do
    subdir="$(basename "$(dirname "$file")")"
    extension="${file##*.}"
    new_name="$OUTPUT_DIR/${subdir}.${extension}"
    cp "$file" "$new_name"
done
