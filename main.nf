#!/usr/bin/env nextflow
nextflow.enable.dsl=2

// Parameters with new defaults matching your structure
params.samples                 = null
params.train_feature_matrix    = "lasso_training_data/feature_matrix_train.csv"
params.model                  = "model_artifacts/lasso_model.joblib"
params.scaler                 = "model_artifacts/scaler.joblib"
params.target                 = null
params.reference_fasta        = "raw_data/NC_045512.2_sequence.fasta"
params.identity_thresh       = 92.0
params.outdir                = "analysis_output"

workflow {
    if (!params.samples) error "Missing --samples parameter"

    Channel.fromPath(params.samples)
           .ifEmpty { error "sample not found: ${params.samples}" }
           .set { sample_ch }

    preprocess(sample_ch) | collapse_predict
}

process preprocess {
    publishDir params.outdir, mode: 'copy'
    tag { sample -> sample }

    input:
    path sample

    output:
    tuple path("new_prep/variant_binary_matrix.csv"), path("new_prep/aligned_filtered.fasta")

    script:
    """
    python3 ${workflow.projectDir}/scripts/preprocess_all.py \
      --samples ${sample} \
      --reference-fasta ${workflow.projectDir}/${params.reference_fasta} \
      --identity-threshold ${params.identity_thresh} \
      --out-dir new_prep
    """
}

process collapse_predict {
    publishDir params.outdir, mode: 'copy'

    input:
    tuple path(variant_matrix), path(aligned_filtered)

    output:
    path "final_predictions/predictions.csv"
    path "final_predictions/collapsed_feature_matrix.csv"
    path "final_predictions/metrics.txt" optional true

    script:
    def target_arg = params.target ? "--target ${params.target}" : ""
    """
    python3 ${workflow.projectDir}/scripts/collapse_and_predict.py \
      --variant-matrix ${variant_matrix} \
      --aligned-fasta ${aligned_filtered} \
      --reference-id NC_045512.2 \
      --train-feature-matrix ${workflow.projectDir}/${params.train_feature_matrix} \
      --model ${workflow.projectDir}/${params.model} \
      --scaler ${workflow.projectDir}/${params.scaler} \
      ${ target_arg } \
      --out-dir final_predictions
    """
}
