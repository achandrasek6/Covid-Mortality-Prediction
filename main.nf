#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

// Parameters
params.samples = null
params.reference_fasta = null
params.train_feature_matrix = null
params.model = null
params.scaler = null
params.chunk_size = 10
params.identity_thresh = 92.0
params.outdir = null
params.verbose = false  // New verbosity parameter

def python_cmd = "python3"

workflow {
    // Parameter validation
    if (!params.samples) error "Please set --samples"
    if (!params.reference_fasta) error "Please set --reference_fasta"
    if (!params.train_feature_matrix) error "Please set --train_feature_matrix"
    if (!params.model) error "Please set --model"
    if (!params.scaler) error "Please set --scaler"
    if (!params.outdir) error "Please set --outdir (output directory is required)"

    // Always show where outputs are being written
    println "[INFO] Writing outputs to: ${params.outdir}"

    // Show additional startup info only if verbose
    if (params.verbose) {
        println "[INFO] Chunk size: ${params.chunk_size}"
        println "[INFO] Identity threshold: ${params.identity_thresh}%"
    }

    // Stage static artifacts as channels
    reference_fasta_ch = Channel.fromPath(params.reference_fasta)
                               .ifEmpty { error "Cannot find reference FASTA: ${params.reference_fasta}" }

    train_feature_matrix_ch = Channel.fromPath(params.train_feature_matrix)
                                    .ifEmpty { error "Cannot find train feature matrix: ${params.train_feature_matrix}" }

    model_ch = Channel.fromPath(params.model)
                     .ifEmpty { error "Cannot find model: ${params.model}" }

    scaler_ch = Channel.fromPath(params.scaler)
                      .ifEmpty { error "Cannot find scaler: ${params.scaler}" }

    // Stage Python scripts
    preprocess_script_ch = Channel.fromPath("${workflow.projectDir}/scripts/preprocess_all.py")
                                 .ifEmpty { error "Cannot find preprocess_all.py script" }

    predict_script_ch = Channel.fromPath("${workflow.projectDir}/scripts/collapse_and_predict.py")
                              .ifEmpty { error "Cannot find collapse_and_predict.py script" }

    // Create channel for multiple FASTA files
    fasta_files_ch = Channel.fromPath(params.samples)
                           .ifEmpty { error "Cannot find FASTA files: ${params.samples}" }
                           .map { fasta ->
                               def basename = fasta.getName().replaceFirst(/\.[^.]+$/, '')
                               tuple(basename, fasta)
                           }

    // Split each FASTA file into chunks
    chunks_ch = fasta_files_ch
                  .flatMap { basename, fasta ->
                      def chunks = fasta.splitFasta(by: params.chunk_size, file: true)
                      def chunkList = []
                      chunks.eachWithIndex { chunk, index ->
                          def chunkName = "${basename}.${index + 1}"
                          chunkList.add(tuple(basename, chunkName, chunk))
                      }
                      return chunkList
                  }

    // Preprocess chunks
    preproc_ch = preprocessChunk(
        chunks_ch,
        reference_fasta_ch.first(),
        preprocess_script_ch.first()
    )

    // Predict on chunks
    preds_ch = predictChunk(
        preproc_ch.map { basename, chunkName, mat, aln, sum ->
            tuple(basename, mat, aln, chunkName)
        },
        train_feature_matrix_ch.first(),
        model_ch.first(),
        scaler_ch.first(),
        predict_script_ch.first()
    )

    // Group results by sample
    preds_grouped = preds_ch.map { basename, pred_file ->
                                tuple(basename, pred_file)
                            }
                            .groupTuple()

    failures_grouped = preproc_ch.map { basename, chunkName, mat, aln, sum ->
                                      tuple(basename, sum)
                                  }
                                 .groupTuple()

    // Merge results
    mergePredictions(preds_grouped)

    // Process failures and show summary if verbose
    failure_results = mergeFailures(failures_grouped)

    // Display failure summary only if verbose mode is enabled
    if (params.verbose) {
        failure_results.subscribe { basename, failure_file ->
            if (failure_file.exists() && failure_file.size() > 0) {
                def lines = failure_file.readLines()
                if (lines.size() > 1) {
                    println "[INFO] ✗ Found ${lines.size() - 1} failures for sample: ${basename}"
                } else {
                    println "[INFO] ✓ No failures found for sample: ${basename}"
                }
            } else {
                println "[INFO] ✓ No failures found for sample: ${basename}"
            }
        }
    }
}

process preprocessChunk {
    tag { chunkName }
    errorStrategy 'ignore'
    conda '/root/miniconda3/envs/covid-lasso-pipeline'

    input:
    tuple val(basename), val(chunkName), path(chunkFasta)
    path reference_fasta
    path preprocess_script

    output:
    tuple val(basename), val(chunkName),
          path("${chunkName}_variant_binary_matrix.csv"),
          path("${chunkName}_aligned_filtered.fasta"),
          path("${chunkName}_summary.tsv")

    script:
    """
    # Use absolute paths to ensure the script can find the files
    CHUNK_FASTA_ABS=\$(readlink -f ${chunkFasta})
    REF_FASTA_ABS=\$(readlink -f ${reference_fasta})
    SCRIPT_ABS=\$(readlink -f ${preprocess_script})

    ${python_cmd} \$SCRIPT_ABS \
      --samples            \$CHUNK_FASTA_ABS \
      --reference-fasta    \$REF_FASTA_ABS \
      --identity-threshold ${params.identity_thresh} \
      --out-dir            ${chunkName}_pre

    mv ${chunkName}_pre/variant_binary_matrix.csv   ${chunkName}_variant_binary_matrix.csv
    mv ${chunkName}_pre/aligned_filtered.fasta      ${chunkName}_aligned_filtered.fasta
    mv ${chunkName}_pre/identity_summary.tsv        ${chunkName}_summary.tsv
    """
}

process predictChunk {
    tag { chunkName }
    conda '/root/miniconda3/envs/covid-lasso-pipeline'

    input:
    tuple val(basename), path(variantMatrix), path(alignedFasta), val(chunkName)
    path train_feature_matrix
    path model
    path scaler
    path predict_script

    output:
    tuple val(basename), path("predictions_${chunkName}.csv")

    script:
    """
    ${python_cmd} ${predict_script} \
      --variant-matrix       ${variantMatrix} \
      --aligned-fasta        ${alignedFasta} \
      --reference-id         NC_045512.2 \
      --train-feature-matrix ${train_feature_matrix} \
      --model                ${model} \
      --scaler               ${scaler} \
      --out-dir              ${chunkName}_pred

    mv ${chunkName}_pred/predictions.csv predictions_${chunkName}.csv
    """
}

process mergePredictions {
    tag { basename }
    publishDir "${params.outdir}/${basename}", mode: 'copy'

    input:
    tuple val(basename), path(predsFiles)

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
    publishDir "${params.outdir}/${basename}", mode: 'copy', saveAs: { filename ->
        // Check if file has actual failures (more than just header)
        def file = new File("${task.workDir}/${filename}")
        if (file.exists()) {
            def lines = file.readLines()
            if (lines.size() > 1) {
                return filename  // Publish if there are failures
            }
        }
        return null  // Don't publish if no failures
    }

    input:
    tuple val(basename), path(summaryFiles)

    output:
    tuple val(basename), path("failures.csv")

    script:
    """
    # Create failures file with header
    echo "sample" > failures.csv

    # Extract failures from summary files
    tr -d '\\r' < ${summaryFiles.join(' ')} \\
      | awk -F'\\t' 'FNR>1 && \$3 ~ /REJECT/ { print \$1 }' \\
      >> failures.csv
    """
}