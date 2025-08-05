#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

// Parameters
params.samples              ?: error("Please set --samples")
params.reference_fasta      ?: error("Please set --reference_fasta")
params.train_feature_matrix ?: error("Please set --train_feature_matrix")
params.model                ?: error("Please set --model")
params.scaler               ?: error("Please set --scaler")
params.outdir               = params.outdir ?: 'analysis_output'
params.chunk_size           = params.chunk_size ?: 10
params.identity_thresh      = params.identity_thresh ?: 92.0

def venv = "${workflow.projectDir}/myenv/bin/python3"

workflow {
  Channel
    .fromPath(params.samples)
    .ifEmpty { error "Cannot find FASTA: ${params.samples}" }
    .splitFasta(by: params.chunk_size, file: true)
    .map { fasta -> tuple(fasta, fasta.baseName) }
    .set { chunks_ch }

  preprocessChunk(chunks_ch)
    .set { preproc_ch }

  predictChunk(preproc_ch)
    .set { preds_ch }

  mergePredictions(preds_ch.collect())
}


process preprocessChunk {
  tag "$chunkName"

  input:
    tuple path(chunkFasta), val(chunkName)

  output:
    tuple \
      path("${chunkName}_variant_binary_matrix.csv"), \
      path("${chunkName}_aligned_filtered.fasta"), \
      val(chunkName)

  script:
  """
  ${venv} ${workflow.projectDir}/scripts/preprocess_all.py \
    --samples         ${chunkFasta} \
    --reference-fasta ${params.reference_fasta} \
    --identity-threshold ${params.identity_thresh} \
    --out-dir         ${chunkName}_pre

  mv ${chunkName}_pre/variant_binary_matrix.csv ${chunkName}_variant_binary_matrix.csv
  mv ${chunkName}_pre/aligned_filtered.fasta  ${chunkName}_aligned_filtered.fasta
  """
}


process predictChunk {
  tag "$chunkName"

  input:
    tuple path(variantMatrix), path(alignedFasta), val(chunkName)

  output:
    path "predictions_${chunkName}.csv"

  script:
  """
  ${venv} ${workflow.projectDir}/scripts/collapse_and_predict.py \
    --variant-matrix       ${variantMatrix} \
    --aligned-fasta        ${alignedFasta} \
    --reference-id         NC_045512.2 \
    --train-feature-matrix ${workflow.projectDir}/${params.train_feature_matrix} \
    --model                ${workflow.projectDir}/${params.model} \
    --scaler               ${workflow.projectDir}/${params.scaler} \
    --out-dir              ${chunkName}_pred

  mv ${chunkName}_pred/predictions.csv predictions_${chunkName}.csv
  """
}


process mergePredictions {
  publishDir "${params.outdir}", mode: 'copy'

  input:
    path predsFiles

  output:
    path "all_predictions.csv"

  script:
  """
  awk -F',' '
    # Only keep one header, and rename it to “predicted_cfr_fraction”
    FNR==1 {
      if (NR==1) { print "sample,predicted_cfr_fraction" }
      next
    }
    # Drop reference-genome rows
    \$1 == "NC_045512.2" { next }
    # Print everything else unchanged
    { print }
  ' ${predsFiles.join(' ')} > all_predictions.csv
  """
}
