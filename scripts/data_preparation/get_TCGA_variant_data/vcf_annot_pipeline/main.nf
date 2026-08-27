#!/usr/bin/env nextflow

nextflow.enable.dsl=2

params.vcfs_path = "/scratch/10900/aksh/TCGA_Raw_VCFs"  // directory with VCF files
params.vcfs_list = "$projectDir/vcfs_list.txt"
params.manifest = "$projectDir/gdc_manifest.2025-10-30.203056.txt"
params.sample_sheet = "$projectDir/gdc_sample_sheet.2025-11-04.tsv"
params.annovar = "$projectDir/annovar"
params.logdir = "$projectDir/download_logs"
params.outdir = "$projectDir/../../data/tcga_data_annotated"

// Step 1: Get variants and patient data
process GET_PATIENT_VARIANT_DATA {
    tag "$vcf_name"

    maxForks 25

    input:
    val vcf_name

    output:
    path "${vcf_name.replace('.vcf', '.tsv')}", emit: tsv

    script:
    """
    python "$projectDir/1.1_get_patient_variant_data.py" \
        --vcf ${params.vcfs_path}/${vcf_name} \
        --manifest ${params.manifest} \
        --sample_sheet ${params.sample_sheet}
    """
}

// Step 2: Annotate variants using Annovar
process ANNOVAR {
    tag "$tsv_file"

    maxForks 25

    publishDir "${params.outdir}", mode: 'copy', overwrite: true

    input:
    path tsv_file

    output:
    path "${tsv_file.baseName}_annot.tsv", emit: annotated_tsv

    script:
    """
    python "$projectDir/1.2_annovar_annotation.py" \
        --mut_file $tsv_file \
        --annovar_dir ${params.annovar} 
    """
}

workflow {
    // read the list of VCF filenames (one per line)
    file_ids_ch = Channel
        .fromPath(params.vcfs_list)
        .splitText()
        .map { it.trim() }
        .filter { it } // remove empty lines

    variant_data = GET_PATIENT_VARIANT_DATA(file_ids_ch)
    annotated_data = ANNOVAR(variant_data.tsv)
}
