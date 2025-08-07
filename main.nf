#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

// Parameters
params.samples              ?: error("Please set --samples")
params.reference_fasta      ?: error("Please set --reference_fasta")
params.train_feature_matrix ?: error("Please set --train_feature_matrix")
params.model                ?: error("Please set --model")
params.scaler               ?: error("Please set --scaler")
params.chunk_size           = params.chunk_size ?: 10
params.identity_thresh      = params.identity_thresh ?: 92.0

// Derive outdir from FASTA basename
def base = file(params.samples).getName().replaceFirst(/\.[^.]+$/, '')
params.outdir = "${base}_out"
println "[INFO] Writing merged outputs to: ${params.outdir}"

def venv = "${workflow.projectDir}/myenv/bin/python3"

workflow {
    // 1) Split and preprocess
    chunks_ch   = Channel.fromPath(params.samples)
                        .ifEmpty { error "Cannot find FASTA: ${params.samples}" }
                        .splitFasta(by: params.chunk_size, file: true)
                        .map { fasta -> tuple(fasta.baseName, fasta) }
    preproc_ch  = preprocessChunk(chunks_ch)

    // 2) Predict on every chunk
    preds_ch    = predictChunk(
                    preproc_ch.map { name, mat, aln, sum ->
                        tuple(mat, aln, name)
                    }
                  )

    // 3) Merge ALL chunk‐level predictions into one CSV
    mergePredictions(preds_ch.collect())

    // 4) Merge ALL chunk‐level failures into one CSV
    mergeFailures(
      preproc_ch
        .map { name, mat, aln, sum -> sum }
        .collect()
    )
}


process preprocessChunk {
    tag { name }
    errorStrategy 'ignore'

    input:
      tuple val(name), path(chunkFasta)

    output:
      tuple val(name),
            path("${name}_variant_binary_matrix.csv"),
            path("${name}_aligned_filtered.fasta"),
            path("${name}_summary.tsv")

    script:
    """
    ${venv} ${workflow.projectDir}/scripts/preprocess_all.py \
      --samples            ${chunkFasta} \
      --reference-fasta    ${params.reference_fasta} \
      --identity-threshold ${params.identity_thresh} \
      --out-dir            ${name}_pre

    mv ${name}_pre/variant_binary_matrix.csv   ${name}_variant_binary_matrix.csv
    mv ${name}_pre/aligned_filtered.fasta      ${name}_aligned_filtered.fasta
    mv ${name}_pre/identity_summary.tsv        ${name}_summary.tsv
    """
}


process predictChunk {
    tag { name }

    input:
      tuple path(variantMatrix), path(alignedFasta), val(name)

    output:
      path("predictions_${name}.csv")

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
    publishDir params.outdir, mode: 'copy'

    input:
      path predsFiles     // List<Path> of all predictions_*.csv

    output:
      path "all_predictions.csv"

    script:
    """
    awk -F',' '
      FNR==1 { if (NR==1) print "sample,predicted_cfr_fraction"; next }
      \$1=="NC_045512.2" { next }
      { print }
    ' ${predsFiles.join(' ')} > all_predictions.csv
    """
}


process mergeFailures {
    publishDir params.outdir, mode: 'copy'

    input:
      path summaryFiles  // List<Path> of all *_summary.tsv

    output:
      path "all_failures.csv"

    script:
    """
    echo "sample" > all_failures.csv
    tr -d '\\r' < ${summaryFiles.join(' ')} \\
      | awk -F'\\t' 'FNR>1 && \$3 ~ /REJECT/ { print \$1 }' \\
      >> all_failures.csv
    """
}
