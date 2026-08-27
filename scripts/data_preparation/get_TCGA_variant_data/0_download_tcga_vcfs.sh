#!/bin/bash
#SBATCH --job-name=cancer_downloads
#SBATCH --chdir=/work/10900/aksh/ls6/cancer_mutation_model
#SBATCH --output=/work/10900/aksh/ls6/logs/vcf_downloads.out
#SBATCH --error=/work/10900/aksh/ls6/logs/vcf_downloads.err
#SBATCH --time=48:00:00
#SBATCH --partition=gg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -A BCS25073
#SBATCH -t 48:00:00 

GDC_CLIENT="scripts/download_tcga_vcfs/src/gdc-client/bin/gdc-client" # update path to your gdc-client setup
MANIFEST_FILE="scripts/download_tcga_vcfs/src/gdc_manifest_vcf.txt" # update path to your manifest file
TOKEN_FILE="scripts/download_tcga_vcfs/src/gdc-user-token.txt" # update path to your token file

# output directory for VCF files
VCF_DATA="/corral/utexas/BCS25073/igs_group/group_resources/TCGA"

# create log and output directory if it doesn’t exist
mkdir -p src/logs
mkdir -p "$VCF_DATA"

# Activate conda environment
source /home1/10900/aksh/miniforge3/bin/activate /home1/10900/aksh/miniforge3/envs/cancer-model

# Download VCF files using GDC client
./${GDC_CLIENT} download -m ${MANIFEST_FILE} -t ${TOKEN_FILE}

# unzip all vcf gunzip files if not already unzipped
for gunzip_file in */*.gz; do
    gunzip -d "$gunzip_file"
done

# move all vcf files to data directory and remove subdirectories
mv */*.vcf "$VCF_DATA"
if [ $? -eq 0 ]; then
    for subdir in */; do
        # exclude src directory from deletion
        [[ "$subdir" == "src/" ]] && continue
        rm -r "$subdir"
    done
fi
