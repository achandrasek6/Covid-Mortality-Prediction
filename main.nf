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
    // split input FASTA into chunks
    Channel
        .fromPath(params.samples)
        .ifEmpty { error "Cannot find FASTA: ${params.samples}" }
        .splitFasta(by: params.chunk_size, file: true)
        .map { fasta -> tuple(fasta, fasta.baseName) }
        .set { chunks_ch }

    // preprocess each chunk (captures summary)
    preprocessChunk(chunks_ch)
        .set { preproc_ch }

    // predict on each successfully preprocessed chunk
    predictChunk(preproc_ch.map { matrix, fasta, summary, name -> tuple(matrix, fasta, name) })
        .set { preds_ch }

    // merge all predictions
    mergePredictions(preds_ch.collect())

    // merge all failures from summaries
    mergeFailures(preproc_ch.map { matrix, fasta, summary, name -> summary }.collect())
}

process preprocessChunk {
    tag "$name"
    errorStrategy 'ignore'

    input:
    tuple path(chunkFasta), val(name)

    output:
    tuple \
      path("${name}_variant_binary_matrix.csv"), \
      path("${name}_aligned_filtered.fasta"), \
      path("${name}_summary.tsv"), \
      val(name)

    script:
    """
    # Run the Python preprocessing — it writes the table to ${name}_pre/identity_summary.tsv
    ${venv} ${workflow.projectDir}/scripts/preprocess_all.py \
        --samples            ${chunkFasta} \
        --reference-fasta    ${params.reference_fasta} \
        --identity-threshold ${params.identity_thresh} \
        --out-dir            ${name}_pre

    # Move outputs into the work dir for Nextflow
    mv ${name}_pre/variant_binary_matrix.csv   ${name}_variant_binary_matrix.csv
    mv ${name}_pre/aligned_filtered.fasta      ${name}_aligned_filtered.fasta
    mv ${name}_pre/identity_summary.tsv        ${name}_summary.tsv
    """
}


process predictChunk {
    tag "$name"

    input:
        tuple path(variantMatrix), path(alignedFasta), val(name)

    output:
        path "predictions_${name}.csv"

    script:
    """
    ${venv} ${workflow.projectDir}/scripts/collapse_and_predict.py \
        --variant-matrix       ${variantMatrix} \
        --aligned-fasta        ${alignedFasta} \
        --reference-id         NC_045512.2 \
        --train-feature-matrix ${workflow.projectDir}/${params.train_feature_matrix} \
        --model                ${workflow.projectDir}/${params.model} \
        --scaler               ${workflow.projectDir}/${params.scaler} \
        --out-dir              ${name}_pred

    mv ${name}_pred/predictions.csv predictions_${name}.csv
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
      FNR==1 {
        if (NR==1) { print "sample,predicted_cfr_fraction" }
        next
      }
      \$1=="NC_045512.2" { next }
      { print }
    ' ${predsFiles.join(' ')} > all_predictions.csv
    """
}

process mergeFailures {
  publishDir "${params.outdir}", mode: 'copy'

  input:
    path summaryFiles

  output:
    path "all_failures.csv"

  script:
  """
  #–– Debug dump of every summary to stderr:
  for f in ${summaryFiles.join(' ')}; do
    echo "=== \$f ===" >&2
    tr -d '\\r' < \$f \
      | sed -n '1,10p' \
      | sed 's/\\t/ | /g' >&2
  done >&2

  #–– Now build the CSV of rejects:
  echo "sample" > all_failures.csv
  tr -d '\\r' < ${summaryFiles.join(' ')} \
    | awk -F'\\t' 'FNR>1 && \$3 ~ /REJECT/ { print \$1 }' \
    >> all_failures.csv
  """
}



