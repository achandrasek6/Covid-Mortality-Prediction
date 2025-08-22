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
params.verbose = false

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
    println "[INFO] Chunk size: ${params.chunk_size}"
    println "[INFO] Identity threshold: ${params.identity_thresh}%"

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
    # Debug: Show current environment
    echo "=== DEBUGGING FILE PATHS ==="
    echo "Current directory: \$(pwd)"
    echo "Available files:"
    ls -la
    echo "Python version: \$(python3 --version)"

    # Get absolute paths to ensure the Python script can find files
    CHUNK_FASTA_ABS="\$(pwd)/${chunkFasta}"
    REF_FASTA_ABS="\$(pwd)/${reference_fasta}"
    SCRIPT_ABS="\$(pwd)/${preprocess_script}"

    echo "Absolute paths:"
    echo "  Chunk FASTA: \$CHUNK_FASTA_ABS"
    echo "  Reference FASTA: \$REF_FASTA_ABS"
    echo "  Script: \$SCRIPT_ABS"

    # Verify files exist with absolute paths
    ls -la "\$CHUNK_FASTA_ABS" || echo "ERROR: Chunk FASTA not found at absolute path"
    ls -la "\$REF_FASTA_ABS" || echo "ERROR: Reference FASTA not found at absolute path"
    ls -la "\$SCRIPT_ABS" || echo "ERROR: Script not found at absolute path"

    echo "=== END DEBUGGING ==="

    # Run preprocessing with absolute paths
    echo "Starting preprocessing with absolute paths..."

    python3 "\$SCRIPT_ABS" \
      --samples            "\$CHUNK_FASTA_ABS" \
      --reference-fasta    "\$REF_FASTA_ABS" \
      --identity-threshold ${params.identity_thresh} \
      --out-dir            ${chunkName}_pre

    PYTHON_EXIT_CODE=\$?
    echo "Python script exit code: \$PYTHON_EXIT_CODE"

    if [ \$PYTHON_EXIT_CODE -ne 0 ]; then
        echo "ERROR: Python script failed with exit code \$PYTHON_EXIT_CODE"
        echo "Current directory contents after failure:"
        ls -la
        exit \$PYTHON_EXIT_CODE
    fi

    # Check if output directory was created
    if [ ! -d "${chunkName}_pre" ]; then
        echo "ERROR: Output directory ${chunkName}_pre was not created"
        echo "Current directory contents:"
        ls -la
        exit 1
    fi

    echo "Output directory contents:"
    ls -la ${chunkName}_pre/

    # Move files with error checking
    if [ -f "${chunkName}_pre/variant_binary_matrix.csv" ]; then
        mv ${chunkName}_pre/variant_binary_matrix.csv ${chunkName}_variant_binary_matrix.csv
        echo "Successfully moved variant_binary_matrix.csv"
    else
        echo "ERROR: variant_binary_matrix.csv not found in output directory"
        ls -la ${chunkName}_pre/
        exit 1
    fi

    if [ -f "${chunkName}_pre/aligned_filtered.fasta" ]; then
        mv ${chunkName}_pre/aligned_filtered.fasta ${chunkName}_aligned_filtered.fasta
        echo "Successfully moved aligned_filtered.fasta"
    else
        echo "ERROR: aligned_filtered.fasta not found in output directory"
        ls -la ${chunkName}_pre/
        exit 1
    fi

    if [ -f "${chunkName}_pre/identity_summary.tsv" ]; then
        mv ${chunkName}_pre/identity_summary.tsv ${chunkName}_summary.tsv
        echo "Successfully moved identity_summary.tsv"
    else
        echo "ERROR: identity_summary.tsv not found in output directory"
        ls -la ${chunkName}_pre/
        exit 1
    fi

    echo "=== FINAL OUTPUT CHECK ==="
    ls -la *.csv *.fasta *.tsv
    echo "=== PREPROCESSING COMPLETED SUCCESSFULLY ==="
    """
}

process predictChunk {
    tag { chunkName }

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
    python3 ${predict_script} \
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
    publishDir "${params.outdir}/${basename}", mode: 'copy'

    input:
    tuple val(basename), path(summaryFiles)

    output:
    tuple val(basename), path("failures.csv")

    script:
    """
    # Create failures file with header
    echo "sample" > failures.csv

    # Debug: Show what summary files we have
    echo "=== DEBUGGING FAILURES PROCESSING ==="
    echo "Processing summary files for sample: ${basename}"
    echo "Summary files:"
    ls -la ${summaryFiles.join(' ')}

    # Show content of each summary file
    for file in ${summaryFiles.join(' ')}; do
        echo "Content of \$file:"
        cat "\$file" || echo "Could not read \$file"
        echo "---"
    done
    echo "=== END DEBUGGING ==="

    # Extract failures from summary files
    cat ${summaryFiles.join(' ')} \\
      | tr -d '\\r' \\
      | awk -F'\\t' 'FNR>1 && \$3 ~ /REJECT/ { print \$1 }' \\
      >> failures.csv

    # Debug: Show final failures file
    echo "=== FINAL FAILURES FILE ==="
    echo "Content of failures.csv:"
    cat failures.csv
    echo "Number of lines in failures.csv: \$(wc -l < failures.csv)"
    echo "=== END FINAL FAILURES ==="
    """
}