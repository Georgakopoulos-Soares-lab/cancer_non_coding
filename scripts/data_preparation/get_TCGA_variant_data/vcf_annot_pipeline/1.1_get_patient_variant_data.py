import os
import argparse
import pandas as pd

# These files need to be downloaded from GDC portal and placed in the specified paths
MANIFEST_FILE = "scripts/1_get_TCGA_variant_data/vcf_annot_pipeline/gdc_manifest.2025-10-30.203056.txt"
SAMPLE_SHEET = "scripts/1_get_TCGA_variant_data/vcf_annot_pipeline/gdc_sample_sheet.2025-11-04.tsv"

def get_end(pos, ref):
    if pos == '-':
        return pos + 1
    return pos + len(ref) - 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vcf", type=str, help="Path to the VCF file")
    parser.add_argument("--manifest", type=str, default=MANIFEST_FILE, help="Path to the manifest file from GDC")
    parser.add_argument("--sample_sheet", type=str, default=SAMPLE_SHEET, help="Path to the sample sheet file from GDC")
    parser.add_argument("--output", type=str, default="", help="Directory to save the patient variant data")
    args = parser.parse_args()
    vcf_file = args.vcf
    manifest_file = args.manifest
    sample_sheet_file = args.sample_sheet
    output_dir = args.output
    
    if len(output_dir) > 0 and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # get mapping of file to patient id and cancer type
    manifest_df = pd.read_csv(manifest_file, sep="\t")
    manifest_df.drop_duplicates(inplace=True)
    sample_df = pd.read_csv(sample_sheet_file, sep="\t")
    sample_df.drop_duplicates(inplace=True)
    manifest_df = manifest_df.merge(sample_df, left_on="id", right_on="File ID", how="left")
    manifest_df["Patient_ID"] = manifest_df["Case ID"].str.split(", ").str[0]
    file_patient_map = manifest_df.set_index("id")["Patient_ID"].to_dict()
    file_cancer_map = manifest_df.set_index("id")["Project ID"].to_dict()
    bcr_patient_barcode = file_patient_map[os.path.basename(vcf_file).replace('.vcf', '')]
    cancer = file_cancer_map[os.path.basename(vcf_file).replace('.vcf', '')]

    # process VCF file to extract high quality somatic mutations
    df = pd.read_csv(vcf_file, sep="\t", comment="#", header=None)
    df.columns = ["chrom", "pos", "id", "ref", "alt", "qual", "filter", "info", "format", "tumor", "normal"]
    print("Total variants in VCF:", df.shape)
    df = df[df["filter"] == "PASS"] # high quality
    print("After selecting high confidence somatic mutations:", df.shape)
    if df.shape[0] == 0:
        print("No high confidence somatic mutations found. Exiting.")
    else:
        df["chrom"] = df["chrom"].str.replace("chr", "")
        df["end"] = df.apply(lambda x: get_end(x["pos"], x["ref"]), axis=1)
        df["mutation"] = df.apply(lambda x: f"{x['chrom']}:{x['pos']}-{x['end']}:{x['ref']}:{x['alt']}", axis=1)
        df = df[["mutation"]]
        df["bcr_patient_barcode"] = bcr_patient_barcode
        df["cancer"] = cancer
        df.drop_duplicates(inplace=True)

    if len(output_dir) > 0:
        output_file = os.path.join(output_dir, os.path.basename(vcf_file).replace('.vcf', '.tsv'))
    else:
        output_file = os.path.basename(vcf_file).replace('.vcf', '.tsv')
    df.to_csv(output_file, sep="\t", index=False)
    print(f"Patient variant data saved to {output_file} with {df.shape[0]} variants.")
    