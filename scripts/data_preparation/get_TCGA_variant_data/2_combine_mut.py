import os
import pandas as pd

# input files/dirs
MANIFEST_FILE = "scripts/1_get_TCGA_variant_data/vcf_annot_pipeline/gdc_manifest.2025-10-30.203056.txt"
SAMPLE_SHEET = "scripts/1_get_TCGA_variant_data/vcf_annot_pipeline/gdc_sample_sheet.2025-11-04.tsv"
TCGA_VAR = "data/tcga_data_annotated"

# output files/dirs
CONSOLIDATED_TCGA_VAR = "data/tcga_patient_variants"

if not os.path.exists(CONSOLIDATED_TCGA_VAR):
    os.makedirs(CONSOLIDATED_TCGA_VAR)

def get_combined_var(tsv_files):
    combined_df = pd.DataFrame()
    for i, file in enumerate(tsv_files):
        df = pd.read_csv(file, sep="\t")
        if df.empty:
            continue
        combined_df = pd.concat([combined_df, df], axis=0)
    combined_df.drop_duplicates(inplace=True)
    return combined_df

sample_df = pd.read_csv(SAMPLE_SHEET, sep="\t")
sample_df.drop_duplicates(inplace=True)
sample_df["Patient_ID"] = sample_df["Case ID"].str.split(", ").str[0]
sample_df = sample_df[["File ID", "Patient_ID"]]
# dictionary with patient IDs as keys and list of file IDs as values
patient_file = sample_df.groupby("Patient_ID")["File ID"].apply(list).to_dict()
print(f"Found {len(patient_file)} unique patients in sample sheet.")

patients = list(patient_file.keys())
for i, patient in enumerate(patients):
    tsv_files = [f"{TCGA_VAR}/{fname}_annot.tsv" for fname in patient_file[patient]]
    tsv_files = [f for f in tsv_files if os.path.exists(f)]
    combined_df = get_combined_var(tsv_files)
    combined_df.to_csv(os.path.join(CONSOLIDATED_TCGA_VAR, f"{patient}.tsv"), sep="\t", index=False)
    if (i + 1) % 1000 == 0:
        print(f"Combined variants for {i+1} patients...")
