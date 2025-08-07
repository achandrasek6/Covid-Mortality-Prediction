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
params.outdir               = params.outdir ?: "results"

println "[INFO] Writing outputs to: ${params.outdir}"

// Use mamba environment instead of venv
def python_cmd = "python3"

workflow {
    // 1) Create channel for multiple FASTA files
    fasta_files_ch = Channel.fromPath(params.samples)
                           .ifEmpty { error "Cannot find FASTA files: ${params.samples}" }
                           .map { fasta ->
                               def basename = fasta.getName().replaceFirst(/\.[^.]+$/, '')
                               tuple(basename, fasta)
                           }

    // 2) Split each FASTA file into chunks and preprocess
    chunks_ch = fasta_files_ch
                  .flatMap { basename, fasta ->
                      fasta.splitFasta(by: params.chunk_size, file: true)
                           .collect { chunk -> tuple(basename, chunk.baseName, chunk) }
                  }

    preproc_ch = preprocessChunk(chunks_ch)

    // 3) Predict on every chunk
    preds_ch = predictChunk(
                 preproc_ch.map { basename, name, mat, aln, sum ->
                     tuple(basename, mat, aln, name)
                 }
               )

    // 4) Group predictions and failures by original FASTA file
    preds_grouped = preds_ch.map { basename, pred_file -> tuple(basename, pred_file) }
                            .groupTuple()

    failures_grouped = preproc_ch.map { basename, name, mat, aln, sum -> tuple(basename, sum) }
                                 .groupTuple()

    // 5) Merge predictions per FASTA file
    mergePredictions(preds_grouped)

    // 6) Merge failures per FASTA file
    mergeFailures(failures_grouped)
}


process preprocessChunk {
    tag { "${basename}_${name}" }
    errorStrategy 'ignore'
    conda '/root/miniconda3/envs/covid-lasso-pipeline'

    input:
      tuple val(basename), val(name), path(chunkFasta)

    output:
      tuple val(basename), val(name),
            path("${name}_variant_binary_matrix.csv"),
            path("${name}_aligned_filtered.fasta"),
            path("${name}_summary.tsv")

    script:
    """
    ${python_cmd} ${workflow.projectDir}/scripts/preprocess_all.py \
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
    tag { "${basename}_${name}" }
    conda '/root/miniconda3/envs/covid-lasso-pipeline'

    input:
      tuple val(basename), path(variantMatrix), path(alignedFasta), val(name)

    output:
      tuple val(basename), path("predictions_${name}.csv")

    script:
    """
    ${python_cmd} ${workflow.projectDir}/scripts/collapse_and_predict.py \
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
    tag { basename }
    publishDir "${params.outdir}/${basename}", mode: 'copy'

    input:
      tuple val(basename), path(predsFiles)     // List<Path> of all predictions_*.csv for this basename

    output:
      tuple val(basename), path("predictions.csv")

    script:
    """
    awk -F',' '
      FNR==1 { if (NR==1) print "sample,predicted_cfr_fraction"; next }
      \$1=="NC_045512.2" { next }
      { print }
    ' ${predsFiles.join(' ')} > predictions.csv
    """
}


process mergeFailures {
    tag { basename }
    publishDir "${params.outdir}/${basename}", mode: 'copy'

    input:
      tuple val(basename), path(summaryFiles)  // List<Path> of all *_summary.tsv for this basename

    output:
      tuple val(basename), path("failures.csv")

    script:
    """
    echo "sample" > failures.csv
    tr -d '\\r' < ${summaryFiles.join(' ')} \\
      | awk -F'\\t' 'FNR>1 && \$3 ~ /REJECT/ { print \$1 }' \\
      >> failures.csv
    """
}