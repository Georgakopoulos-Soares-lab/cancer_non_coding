import argparse
import os
import pandas as pd

TCGA_PATIENT_LIST = "metadata/patient_list.txt"
CNV_DATA = "data/TCGA/cnv_data"
CNV_SAMPLE_SHEET = "scripts/1.1_download_TCGA_data/cnv/gdc_sample_sheet.2026-02-21.tsv"
GENE_EXP_DATA = "data/TCGA/gene_expression_data"
GENE_EXP_SAMPLE_SHEET = "scripts/1.1_download_TCGA_data/gene_exp/gdc_sample_sheet.2026-02-21.tsv"
DEFAULT_OUTPUT = "metadata/patient_cnv_gene_exp_mapping.tsv"


def _load_patient_list(path):
	with open(path, "r") as f:
		return [line.strip() for line in f if line.strip()]


def _normalize_patient_id(case_id):
	if pd.isna(case_id):
		return ""
	return str(case_id).split(", ")[0]


def _pick_preferred_file(df):
	if df.empty:
		return ""
	df = df.copy()
	df["_has_tumor"] = df["Tissue Type"].fillna("").str.contains("Tumor")
	df["_has_01"] = df["Sample ID"].fillna("").str.contains("-01")
	df = df.sort_values(by=["_has_tumor", "_has_01", "File ID"], ascending=[False, False, True])
	return df.iloc[0]["File ID"]


def _build_file_map(sample_sheet_path):
	df = pd.read_csv(sample_sheet_path, sep="\t")
	df.drop_duplicates(inplace=True)
	df["Patient_ID"] = df["Case ID"].apply(_normalize_patient_id)
	return df


def build_mapping(patient_list, cnv_sample_sheet, gene_exp_sample_sheet, cnv_dir, gene_exp_dir):
	cnv_df = _build_file_map(cnv_sample_sheet)
	gene_df = _build_file_map(gene_exp_sample_sheet)

	cnv_grouped = cnv_df.groupby("Patient_ID", dropna=False)
	gene_grouped = gene_df.groupby("Patient_ID", dropna=False)

	rows = []
	for patient_id in patient_list:
		cnv_file = ""
		gene_file = ""

		if patient_id in cnv_grouped.groups:
			cnv_file = _pick_preferred_file(cnv_grouped.get_group(patient_id))

		if patient_id in gene_grouped.groups:
			gene_file = _pick_preferred_file(gene_grouped.get_group(patient_id))

		rows.append(
			{
				"patient_id": patient_id,
				"cnv_file": f"{os.path.join(cnv_dir, cnv_file)}.txt" if cnv_file else "",
				"gene_exp_file": f"{os.path.join(gene_exp_dir, gene_file)}.tsv" if gene_file else "",
			}
		)

	return pd.DataFrame(rows)


def _log_missing(kind, patient_ids):
	if not patient_ids:
		print(f"No missing {kind} files.")
		return
	print(f"Missing {kind} files ({len(patient_ids)}):")
	print(patient_ids[:5]) # Print only the first 5 for brevity


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Create patient-to-file mapping for CNV and gene expression data.")
	parser.add_argument("--patient-list", default=TCGA_PATIENT_LIST)
	parser.add_argument("--cnv-sample-sheet", default=CNV_SAMPLE_SHEET)
	parser.add_argument("--gene-exp-sample-sheet", default=GENE_EXP_SAMPLE_SHEET)
	parser.add_argument("--cnv-dir", default=CNV_DATA)
	parser.add_argument("--gene-exp-dir", default=GENE_EXP_DATA)
	parser.add_argument("--output", default=DEFAULT_OUTPUT)
	args = parser.parse_args()

	patients = _load_patient_list(args.patient_list)
	mapping_df = build_mapping(
		patients,
		args.cnv_sample_sheet,
		args.gene_exp_sample_sheet,
		args.cnv_dir,
		args.gene_exp_dir,
	)

	output_dir = os.path.dirname(args.output)
	if output_dir:
		os.makedirs(output_dir, exist_ok=True)
	mapping_df.to_csv(args.output, sep="\t", index=False)
	print(f"Mapping file written to {args.output} with {mapping_df.shape[0]} patients.")

	missing_cnv = mapping_df[mapping_df["cnv_file"] == ""]["patient_id"].tolist()
	missing_gene = mapping_df[mapping_df["gene_exp_file"] == ""]["patient_id"].tolist()
	_log_missing("CNV", missing_cnv)
	_log_missing("gene expression", missing_gene)