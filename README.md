# SARS‑CoV‑2 Case‑Fatality‑Rate (CFR) Prediction Pipeline

Reproducible, end‑to‑end workflow to: (1) fetch genomes, (2) align to Wuhan‑Hu‑1 (NC\_045512.2), (3) encode binary mutation features, (4) train an L1‑regularized Lasso regressor to predict global CFR, (5) run robustness checks (label permutations, feature shuffles, ablations), (6) produce explanations (SHAP/LIME), and (7) batch‑score new genomes. An optional DNABERT fine‑tuning module provides a deep‑learning baseline/ensemble.

---

## Table of Contents

* [Overview](#overview)
* [Directory Layout](#directory-layout)
* [Environment](#environment)
* [Quickstart](#quickstart)
* [CLI by Stage](#cli-by-stage)

  * [0) Get the reference & annotations](#0-get-the-reference--annotations)
  * [1) Subsample FASTA for quick iteration (optional)](#1-subsample-fasta-for-quick-iteration-optional)
  * [2) Full preprocessing: align → filter → variant matrix](#2-full-preprocessing-align--filter--variant-matrix)
  * [3) Train & evaluate Lasso](#3-train--evaluate-lasso)
  * [4) Robustness: negative controls & ablations](#4-robustness-negative-controls--ablations)
  * [5) Model explanations (SHAP & LIME)](#5-model-explanations-shap--lime)
  * [6) Collapse and predict on new genomes](#6-collapse-and-predict-on-new-genomes)
  * [7) Visualizations](#7-visualizations)
  * [8) DNABERT baseline (optional)](#8-dnabert-baseline-optional)
* [Nextflow entrypoint](#nextflow-entrypoint)
* [Docker image](#docker-image)
* [Reproducibility](#reproducibility)
* [Troubleshooting](#troubleshooting)
* [License](#license)

---

## Overview

This repository organizes a classical‑ML pipeline for CFR prediction using interpretable, position‑specific mutation features derived from a multiple‑sequence alignment. Utilities are provided for data subsampling, feature inspection, robustness checks, and visualization. A TensorFlow‑based DNABERT fine‑tune is included as an optional baseline/ensemble.

```mermaid
flowchart LR
  A["Fetch genomes / provide FASTA"] --> B["Reference: NC_045512.2"]
  B --> C["MAFFT alignment"]
  C --> D["Filter by percent identity"]
  D --> E["Binary variant feature matrix"]
  E --> F["Lasso training + artifacts"]
  F --> G["Explain (SHAP/LIME)"]
  E --> H["Variant heatmap"]
  D --> I["Collapse features for new genomes"]
  F --> I
  I --> J["Predict CFR"]

---

## Directory Layout

```
project/
├─ raw_data/
│  ├─ NC_045512.2_sequence.fasta            # reference genome (created by Ref_Seq_Import.py)
│  └─ NC_045512.2_gene_annotations.txt      # gene coordinates (created by Ref_Seq_Import.py)
├─ transformed_data/
│  ├─ variant_samples.fasta                 # fetched or provided
│  ├─ variant_samples_small.fasta           # optional subsample for dev
│  └─ aligned.fasta                         # MAFFT output (if using simple path)
├─ preprocessed_full/
│  ├─ aligned_raw.fasta                     # alignment (reference + samples)
│  ├─ aligned_filtered.fasta                # filtered by % identity, reference first
│  ├─ identity_summary.tsv                  # pass/reject report
│  ├─ rejected/                             # per‑sample FASTAs that failed threshold
│  └─ variant_binary_matrix.csv             # binary mutation matrix
├─ lasso_training_data/
│  ├─ feature_matrix_train.csv              # training split
│  └─ feature_matrix_test.csv               # test split
├─ model_artifacts/
│  ├─ lasso_model.joblib
│  └─ scaler.joblib
├─ explanations/                            # SHAP/LIME HTMLs + CSVs/PNGs
├─ collapsed_prediction/                    # collapsed features + predictions
├─ figures/                                 # visualizations (e.g., elbow/heatmap)
├─ scripts/                                 # all Python scripts referenced below
│  ├─ NCBI_variant_sampling.py
│  ├─ Ref_Seq_Import.py
│  ├─ Alignment.py
│  ├─ preprocess_all.py
│  ├─ Build_matrix.py
│  ├─ ML_model.py
│  ├─ ML_model_user_CLI.py
│  ├─ Regularization.py
│  ├─ Selected_Features.py
│  ├─ Variant_feature_heatmap.py
│  ├─ neg_ctrls_ablations.py
│  ├─ explain_lasso.py
│  ├─ collapse_and_predict.py
│  ├─ Examine_matrix.py
│  └─ extract_features.py
├─ Dockerfile.lasso                         # Docker image for Lasso pipeline
├─ environment.yml                          # Conda environment
├─ requirements.txt                         # pip requirements (alternative to Conda)
├─ main.nf                                  # Nextflow pipeline entrypoint
└─ nextflow.config                          # Nextflow profiles/executor/config
```

> Some scripts assume relative paths (e.g., `../raw_data`). Run from `scripts/` or adjust paths.

---

## Environment

### Conda

```bash
conda env create -f environment.yml
conda activate covid-lasso-pipeline
```

### Python (pip)

```bash
pip install -r requirements.txt
```

---

## Quickstart

1. **Fetch reference & annotations**

   ```bash
   (cd scripts && python Ref_Seq_Import.py)
   ```
2. **(Optional) Subsample sequences for quick iteration**

   ```bash
   (cd scripts && python subsample_fasta.py \
       -i ../transformed_data/variant_samples.fasta \
       -o ../transformed_data/variant_samples_small.fasta \
       -k 100)
   ```
3. **Full preprocessing → variant matrix**

   ```bash
   (cd scripts && python preprocess_all.py \
       --samples ../transformed_data/variant_samples_small.fasta \
       --reference-fasta ../raw_data/NC_045512.2_sequence.fasta \
       --identity-threshold 92 \
       --out-dir ../preprocessed_full)
   ```
4. **Train/evaluate Lasso & save artifacts**

   ```bash
   (cd scripts && python ML_model.py \
       --train-matrix ../lasso_training_data/feature_matrix_train.csv \
       --test-matrix  ../lasso_training_data/feature_matrix_test.csv \
       --alpha 0.000174 \
       --out-dir ../model_artifacts)
   ```
5. **Explain model (SHAP/LIME)**

   ```bash
   (cd scripts && python explain_lasso.py \
       --train_csv ../lasso_training_data/feature_matrix_train.csv \
       --test_csv  ../lasso_training_data/feature_matrix_test.csv \
       --artifacts_dir ../model_artifacts \
       --outdir ../explanations \
       --lime_n 5 --lime_select largest_error --lime_space raw --lime_digits 6)
   ```
6. **Collapse & predict new genomes**

   ```bash
   (cd scripts && python collapse_and_predict.py \
       --variant-matrix ../preprocessed_full/variant_binary_matrix.csv \
       --aligned-fasta ../preprocessed_full/aligned_filtered.fasta \
       --reference-id NC_045512.2 \
       --train-feature-matrix ../lasso_training_data/feature_matrix_train.csv \
       --model ../model_artifacts/lasso_model.joblib \
       --scaler ../model_artifacts/scaler.joblib \
       --out-dir ../collapsed_prediction)
   ```

---

## CLI by Stage

### 0) Get the reference & annotations

Creates `raw_data/NC_045512.2_sequence.fasta` and a gene table.

```bash
(cd scripts && python Ref_Seq_Import.py)
```

### 1) Subsample FASTA for quick iteration (optional)

Deterministic reservoir subsampling for large datasets.

```bash
(cd scripts && python subsample_fasta.py \
    -i ../transformed_data/variant_samples.fasta \
    -o ../transformed_data/variant_samples_small.fasta \
    -k 250 --seed 42)
```

### 2) Full preprocessing: align → filter → variant matrix

Performs MAFFT alignment, reorders with reference first, filters by % identity, writes identity report, and builds the binary mutation matrix. Also collects rejected samples into `preprocessed_full/rejected/`.

```bash
(cd scripts && python preprocess_all.py \
    --samples ../transformed_data/variant_samples_small.fasta \
    --reference-fasta ../raw_data/NC_045512.2_sequence.fasta \
    --identity-threshold 92 \
    --out-dir ../preprocessed_full \
    --mafft-args --thread -1)
```

**Outputs** (under `preprocessed_full/`):

* `aligned_raw.fasta`, `aligned_filtered.fasta`
* `identity_summary.tsv`, `rejected/`
* `variant_binary_matrix.csv`

### 3) Train & evaluate Lasso

Use your prepared `feature_matrix_train/test.csv` or the matrix from preprocessing after your split step.

```bash
(cd scripts && python ML_model.py \
    --train-matrix ../lasso_training_data/feature_matrix_train.csv \
    --test-matrix  ../lasso_training_data/feature_matrix_test.csv \
    --alpha 0.000174 \
    --out-dir ../model_artifacts)
```

Helpers:

* `Selected_Features.py` — list non‑zero‑coef features at a chosen α.
* `Regularization.py` — sweep α for sparsity/accuracy trade‑off (elbow plot in `figures/`).
* `Examine_matrix.py`, `extract_features.py` — explore headers/feature subsets.

### 4) Robustness: negative controls & ablations

Runs label‑permutation controls, feature‑column shuffles (train‑only), and ablations (regex/list/top‑k by |coef|). Writes CSVs + PNGs + summary JSON.

```bash
(cd scripts && python neg_ctrls_ablations.py \
    --train_csv ../lasso_training_data/feature_matrix_train.csv \
    --test_csv  ../lasso_training_data/feature_matrix_test.csv  \
    --target_col "Global CFR" --id_col SampleID \
    --outdir ../explanations/controls_out \
    --use_lassocv --cv_folds 5 \
    --n_label_perm 200 --n_feat_shuffle 100 \
    --ablate_regex "^S_" "^ORF1ab_" \
    --ablate_list ../key_sites.txt \
    --ablate_topk_coef 50 \
    --save_preds)
```

**Key artifacts:** `baseline_metrics.json`, `label_permutations.csv`, `feature_shuffles.csv`, `ablations.csv`, plots, and `summary.json`.

### 5) Model explanations (SHAP & LIME)

Produces SHAP summary plots and LIME local explanations for selected test samples.

```bash
(cd scripts && python explain_lasso.py \
    --train_csv ../lasso_training_data/feature_matrix_train.csv \
    --test_csv  ../lasso_training_data/feature_matrix_test.csv  \
    --artifacts_dir ../model_artifacts \
    --outdir ../explanations \
    --lime_n 5 --lime_select largest_error --lime_space raw --lime_digits 6)
```

### 6) Collapse and predict on new genomes

Maps features from a new alignment to the training feature space and emits predictions.

```bash
(cd scripts && python collapse_and_predict.py \
    --variant-matrix ../preprocessed_full/variant_binary_matrix.csv \
    --aligned-fasta ../preprocessed_full/aligned_filtered.fasta \
    --reference-id NC_045512.2 \
    --train-feature-matrix ../lasso_training_data/feature_matrix_train.csv \
    --model ../model_artifacts/lasso_model.joblib \
    --scaler ../model_artifacts/scaler.joblib \
    --out-dir ../collapsed_prediction)
```

### 7) Visualizations

Variant × Feature heatmap and α‑sweep elbow plot.

```bash
# Heatmap (Variant_feature_heatmap.py)
(cd scripts && python Variant_feature_heatmap.py)
# Elbow plot (Regularization.py)
(cd scripts && python Regularization.py)
```

Artifacts saved under `figures/`.

### 8) DNABERT baseline (optional)

Fine‑tune DNABERT as a deep‑learning regressor and optionally ensemble with Lasso.

```bash
(cd scripts && python finetune_dnabert_cfr_aug.py --help)
```

---

## Nextflow entrypoint

A Nextflow wrapper (`main.nf`) orchestrates the stages above for scalable, parallel execution.

**Typical usage**

```bash
# Local execution
nextflow run main.nf -profile local \
  --samples "transformed_data/variant_samples_small.fasta" \
  --outdir results_nf

# Docker execution (per‑process containers)
nextflow run main.nf -profile docker \
  --samples "transformed_data/variant_samples_small.fasta" \
  --outdir results_nf
```

See `nextflow.config` for available profiles (e.g., `local`, `docker`) and tunables like CPUs/memory, container images, and work directory. Override at runtime with `-with-report`, `-with-trace`, `-with-dag flowchart.png`, and resume with `-resume`.

---

## Docker image

Build a runtime with all dependencies for the Lasso pipeline.

```bash
# build
docker build -f Dockerfile.lasso -t cfr-lasso:latest .
# run (mount project into /work)
docker run --rm -v "$PWD":/work -w /work cfr-lasso:latest \
  python scripts/ML_model.py --help
```

For end‑to‑end runs, combine with `main.nf -profile docker` and mount MAFFT/data volumes as needed.

---

## Reproducibility

* **Environments**: `environment.yml` (Conda) and `requirements.txt` (pip) capture dependencies.
* **Determinism**: scripts accept seeds where relevant; prefer fixed seeds for subsampling and CV.
* **Artifacts**: models, scalers, reports, and plots are written to version‑controlled directories with timestamps.

---

## Troubleshooting

* **MAFFT not found**: ensure it is installed and on `PATH`.
* **Path errors**: some scripts assume execution from `scripts/`; either run from there or adjust relative paths.
* **Nextflow var errors**: confirm `main.nf` input channels match the declared parameters; use `-with-dag` to inspect graph.
* **CSV parsing**: use `,` for CSV and `\t` for TSV outputs; large CSVs can be grepped but consider `awk`/`csvkit` for robustness.

---

## License

MIT.
