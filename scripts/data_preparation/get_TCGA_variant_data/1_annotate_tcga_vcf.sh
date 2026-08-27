#!/bin/bash
#SBATCH --job-name=annot_vcf
#SBATCH --chdir=/work/10900/aksh/ls6/cancer_mutation_model/scripts/1_get_TCGA_variant_data/vcf_annot_pipeline
#SBATCH --output=/work/10900/aksh/ls6/cancer_mutation_model/scripts/1_get_TCGA_variant_data/vcf_annot_pipeline/logs/nf.out
#SBATCH --error=/work/10900/aksh/ls6/cancer_mutation_model/scripts/1_get_TCGA_variant_data/vcf_annot_pipeline/logs/nf.err
#SBATCH --time=48:00:00
#SBATCH --partition=gg
#SBATCH --nodes=1
#SBATCH -A BCS25073

# create log directory if it doesn’t exist
mkdir -p logs
mkdir -p nf_reports

# Activate conda environment
source /home1/10900/aksh/miniforge3/bin/activate /home1/10900/aksh/miniforge3/envs/cancer-model

# extract .vcf.gz files if not already extracted
VCF_DIR="/scratch/10900/aksh/TCGA_Raw_VCFs"
for file in $VCF_DIR/*.vcf.gz; do
    if [ ! -f "${file%.gz}" ]; then
        gunzip -k "$file"
    fi
done

# create a list of VCF files
vcf_list="/scratch/10900/aksh/nf_work/vcf_file_list.txt"
ls /scratch/10900/aksh/TCGA_Raw_VCFs/*.vcf > $vcf_list

# Run Nextflow pipeline
nextflow run main.nf \
    -resume \
    --vcf_list $vcf_list \
    -work-dir /scratch/10900/aksh/nf_work \
    -with-report nf_reports/nf_report.html \
    -with-trace nf_reports/nf_trace.txt \
    -with-timeline nf_reports/nf_timeline.html \
    -with-dag nf_reports/nf_dag.png
