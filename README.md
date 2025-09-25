# SARS-CoV-2 Case-Fatality Rate (CFR) Prediction Pipeline

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/badge/license-Apache--2.0-blue)
![Nextflow](https://img.shields.io/badge/Nextflow-DSL2-orange)
![AWS](https://img.shields.io/badge/AWS-Batch%20%7C%20Fargate-lightgrey)
[![Build & Push](https://github.com/achandrasek6/Covid-Mortality-Prediction/actions/workflows/ecr-push.yml/badge.svg)](https://github.com/achandrasek6/Covid-Mortality-Prediction/actions/workflows/ecr-push.yml)
[![Build & Push (API)](https://github.com/achandrasek6/Covid-Mortality-Prediction/actions/workflows/ecr-push-fastapi.yml/badge.svg)](https://github.com/achandrasek6/Covid-Mortality-Prediction/actions/workflows/ecr-push-fastapi.yml)

A production-ready, reproducible pipeline for predicting COVID-19 variant-specific case fatality rates. Built with **Nextflow DSL2**, containerized with **Docker**, and deployable on **AWS Batch/ECS**.

**Status: v1.1 (Sep 2025) — Core pipeline shipped: Nextflow DSL2 + AWS Batch distributed scoring, interpretable Lasso baseline, robustness & explainability, and a FastAPI microservice on ECS Fargate (ALB). DNABERT trained (standalone); NF/GPU integration next.**

---

## 📖 Documentation

Use these quick links to jump around this README:

- [🔬 Results at a Glance](#-results-at-a-glance)
- [📈 Key Figures](#-key-figures)
- [⚡ TL;DR](#-tldr)
- [📌 Features](#-features)
- [🗂️ Repository Layout](#️-repository-layout)
- [🚀 Local Quickstart](#-local-quickstart)
- [🧩 API quickstart (Fargate)](#-api-quickstart-fargate)
- [🧬 Workflow Overview](#-workflow-overview)
- [⚙️ Productionization](#️-productionization)
- [💻 CLI by Stage](#-cli-by-stage)
- [🧭 Nextflow Entrypoint](#-nextflow-entrypoint)
- [📊 Outputs](#-outputs)
- [📌 Roadmap](#-roadmap)
- [🧰 Troubleshooting](#-troubleshooting)
- [📜 License](#-license)

---

# 🔬 Results at a Glance

**What I built.** A production-ready pipeline that predicts **variant-specific COVID-19 CFR from viral genomes**, orchestrated with **Nextflow DSL2**, containerized with **Docker**, and deployable on **AWS Batch/ECS**. A fine‑tuned **DNABERT** model (transformer on 6‑mer tokens) is trained as a higher‑capacity head; Nextflow integration is planned as an optional GPU stage.

**Accuracy (held-out test, Lasso).** **R² = 0.831**, **RMSE ≈ 0.00194**, **MAE ≈ 0.00050**, **Spearman = 0.804**.

**Why it matters.**

* **Actionable surveillance:** turns raw genomes into **risk scores** for variants—useful for triage, early warning, and prioritizing wet-lab follow-ups.
* **Interpretable by design:** sparse Lasso highlights a compact set of mutation features, so results are explainable to scientists and decision-makers.
* **Built to scale/reproduce:** Nextflow + Docker + AWS mean the same pipeline runs locally and in the cloud with pinned dependencies and clean provenance.

**From metrics to mechanism:** where the signal lives in the genome:

<img src="figures/variant_feature_heatmap.png"
     alt="Per-variant fraction of samples with each mutation-derived feature. Columns are grouped by genomic region (ORF1ab, Spike, Other) with a colored header strip; rows are variants (Alpha–Omicron). WildType excluded."
     width="95%" />

<p style="max-width:900px; margin: 0.75rem auto 0; font-size: 0.95em; line-height:1.45;">
This heatmap shows the fraction of genomes within each lineage carrying each mutation-derived feature. Columns are grouped and ordered by genomic position—left→right: <strong>ORF1ab (replicase)</strong>, <strong>Spike (S)</strong>, then <strong>Other</strong> (N/M/E + accessory); within each region, features also follow genomic order. The visually higher mutation density in Spike likely reflects (i) <em>positive selection</em> for host-entry and immune-escape changes (RBD, NTD “antigenic supersite,” S1/S2 cleavage), (ii) <em>surveillance bias</em>—Spike-focused reporting and curation make S changes more consistently captured, and (iii) <em>tolerance</em>—replication enzymes in ORF1ab are more constrained, so fewer substitutions persist there. Note that “prevalence” here indicates how common a feature is within a variant, not its effect size; downstream ablations/SHAP quantify influence. <em>WildType excluded.</em>
</p>


---

## 📈 Key Figures



### Lasso Model 

#### 1) Overall Performance (stability across resamples)

The distribution of test R² across bootstrap resamples centers around the reported score indicating **stable generalization** rather than a single lucky split.

![Bootstrap Test R² Distribution](figures/bootstrap_r2_histogram.png)



#### 2) Robustness Checks (controls)

Both controls show the model isn’t learning artifacts.

| **Label permutations** | **Feature shuffles** |
| --- | --- |
| <img src="https://github.com/user-attachments/assets/b8fe9a63-c2c9-46c4-aacc-2ff920dbe9b5" alt="Label permutation R² distribution (Lasso baseline)" width="100%"/> | <img src="https://github.com/user-attachments/assets/b91799ee-f531-47d6-98ee-f9bf544c06c9" alt="Feature shuffle R² distribution (Lasso baseline)" width="100%"/> |
| <sub>Shuffle **labels**: breaks the signal; R² histogram is centered **well below 0**, confirming the model isn’t fitting noise.</sub> | <sub>Shuffle **features** in training only: destroys feature structure; test R² **collapses**, showing real dependence on true features.</sub> |



#### 3) What Genes Matter (group ablations)

Removing the **top-50 |coef|** features yields the largest drop (**ΔR² ≈ −0.033**), validating the coefficient ranking. Dropping **Spike (`^S_`)** features (**ΔR² ≈ −0.025**) and **ORF1ab** (**ΔR² ≈ −0.019**) also harms performance—evidence these genes carry real signal.

<img width="1579" height="580" alt="ablations_delta_r2" src="https://github.com/user-attachments/assets/00d77aed-93b2-4e8e-89d3-7969d9b24ad6" />

<h4>4) Model Explainability (SHAP/LIME, Lasso baseline)</h4>

<table>
  <tr>
    <td width="50%">
      <strong>A. SHAP beeswarm (Lasso baseline)</strong><br/>
      <img src="https://github.com/user-attachments/assets/fcaed853-98de-4f72-afe0-d31a499e75b0"
           alt="A: SHAP beeswarm/density — per-sample feature contributions (Lasso baseline)"
           width="100%"/>
      <div><em>Each dot is a sample’s SHAP value (units=CFR). Right = ↑CFR, left = ↓CFR. Color: red=present, blue=absent. Saturation: Dark=high density, light=low density</em></div>
    </td>
    <td width="50%">
      <strong>B. Top-10 mean(|SHAP|) features (Lasso baseline)</strong><br/>
      <img src="https://github.com/user-attachments/assets/fe8cfab1-7e8b-44f6-be10-52edefeda0bc"
           alt="B: Top-10 features by mean(|SHAP|) (Lasso baseline)"
           width="100%"/>
      <div><em>Average absolute SHAP across the test set—taller bars = more influential features.</em></div>
    </td>
  </tr>
</table>



#### C) LIME case studies (per-sample explanations)

| **A. High-error case studies** | **B. Per-sample waterfall (MZ314997.2)** |
| --- | --- |
| <img src="https://github.com/user-attachments/assets/9588f24e-1081-4d97-bc71-7ec5c7f55243" alt="A: LIME case studies — y_true vs y_pred (top-5 labeled)"/> | <img src="https://github.com/user-attachments/assets/65cba188-044e-41bc-a6e7-05443e8a3663" alt="B: MZ314997.2 — Waterfall (Top-5 features → Prediction)"/> |
| **What you’re seeing:** Ten high-error genomes against the identity line; the **top-5 absolute errors are labeled** to spotlight where the model deviates (useful for stress-testing explanations). | **What you’re seeing:** Baseline CFR adjusted by the **five largest local contributions** to reach the prediction; dashed line shows **true CFR**. **Right bars raise** the score; **left bars lower** it. |

> **How LIME complements SHAP:**  
> SHAP shows **global patterns** (which features matter overall and in which direction per sample), while **LIME** zooms into **one sample at a time** with a simple surrogate explaining its specific prediction (great for narrative, QA, and debugging outliers).

**Quick takeaways (Lasso baseline):**
- Spike and ORF1ab sites dominate top importance—consistent with group ablations.  
- Beeswarm directionality highlights which alleles **raise vs. lower** predicted CFR, guiding biological follow-up.  
- LIME helps **audit individual genomes** (especially large-error cases) to ensure the model’s rationale is sensible.

### Language model (DNABERT) — quick summary

* Fine-tuned transformer achieved **RMSE = 0.0046**, about **15% lower error** than Lasso in my study, while capturing **long-range sequence context**. **Recommended when maximum accuracy is needed.**
* Useful when you want maximum accuracy and are OK with GPU/latency trade-offs; Lasso remains the fast, interpretable default.
* Artifacts included: SavedModel/weights and tokenizer; NF GPU integration planned so you can toggle `--use_dnabert true`.

| Model                | Strengths                                                | Trade-offs                             | Best for                                                              |
| -------------------- | -------------------------------------------------------- | -------------------------------------- | --------------------------------------------------------------------- |
| **Lasso (baseline)** | Interpretable coefficients; small, fast; easy to explain | May miss non-linear/long-range effects | Routine surveillance, explainability, bulk scoring                    |
| **DNABERT (deep)**   | Captures context; headroom for accuracy                  | GPU needed; slower; less transparent   | High-stakes analyses, research scenarios where extra accuracy matters |

## ⚡ TL;DR

**Accurate (R² \~0.83), interpretable, and production-ready** genomic risk prediction.
Deep-learning headroom (DNABERT trained) is available; the shipped Lasso baseline already gives strong accuracy with transparent mutation-level insights and real robustness evidence.

<details>
<summary><strong>Bonus: Model parsimony</strong></summary>

An elbow curve shows lasso model performance saturating with a relatively small number of features—useful for **simpler, faster deployments** and easier biological review.

![Elbow: #features vs Test R²](figures/elbow_plot.png)

</details>

---
## 📌 Features
- **End-to-end workflow**: Fetch genomes → align (MAFFT) → build mutation features → train ML models → explain results.
- **Classical ML**: L1-regularized Lasso regression for interpretable, sparse mutation features.
- **Deep learning baseline (trained; NF integration pending)**: DNABERT fine-tuning module (TensorFlow).
- **Robustness checks**: Label permutations, feature shuffles, ablations.
- **Explainability**: SHAP/LIME for mutation-level interpretation.
- **Scalability**: Nextflow orchestration with AWS Batch for parallel execution.
- **Observability**: Centralized logging & metrics with **Amazon CloudWatch** (Logs, Metrics, Alarms, Insights).
- **Productionization**: CI/CD (GitHub Actions → Docker/ECR), MLflow/DVC for versioning (planned), FastAPI service on ECS Fargate (planned), S3 pre-signed I/O + SQS (planned), drift reports (planned).

---

## 🗂️ Repository Layout

```
project/
├─ .github/                          # GitHub config (Actions workflows, etc.)
├─ app/                              # FastAPI service (main.py, requirements.txt)
├─ controls_out/                     # Robustness outputs (label perms, shuffles, ablations)
├─ dnabert_cfr_regressor...          # DNABERT weights + tokenizer artifacts
├─ docker/                           # NF container build context (Dockerfile.lasso lives here)
├─ explanations/                     # SHAP/LIME figures, explanation reports
├─ figures/                          # Visualizations & diagrams used in the README/papers
├─ lasso_training_data/              # Train/test feature matrices for Lasso
├─ model_artifacts/                  # Trained models, scalers, checkpoints
├─ raw_data/                         # Reference genomes & annotations
├─ scripts/                          # Python utilities & CLI entrypoints
├─ test_samples/                     # Small FASTA samples for quick runs
├─ transformed_data/                 # Prepared/subsampled input FASTAs
├─ .dockerignore                     # Build context excludes for Docker
├─ .gitignore                        # Git ignore rules
├─ Dockerfile.api                    # API Container build context
├─ CITATION.cff                      # Citation metadata for the project
├─ README.md                         # This documentation
├─ environment.yml                   # Conda environment spec
├─ requirements.txt                  # pip requirements
├─ main.nf                           # Nextflow pipeline entrypoint
└─ nextflow.config                   # Nextflow profiles & executor configs
```

> Some scripts assume relative paths (e.g., `../raw_data`). Run from `scripts/` or adjust paths.

---

## 🚀 Local Quickstart

### Environment
```bash
conda env create -f environment.yml
conda activate covid-lasso-pipeline
# or
pip install -r requirements.txt
```

### Docker Image
Build a runtime with all dependencies for the Lasso pipeline.
```bash
docker build -f Dockerfile.lasso -t covid-lasso:aws-preprocess-fix-20250822-0339 .
docker run --rm -v "$PWD":/work -w /work covid-lasso:aws-preprocess-fix-20250822-0339 \
  python scripts/ML_model.py --help
```
For end-to-end runs, combine with `main.nf -profile docker` and mount MAFFT/data volumes as needed.

### End-to-end run (Dockerized)
```bash
nextflow run main.nf -profile docker \
  --samples "transformed_data/variant_samples_small.fasta" \
  --outdir results_nf
```
>See `nextflow.config` file for full list of arguements.
### Train & Evaluate Lasso
```bash
python scripts/ML_model_CLI_user.py \
  --train-matrix lasso_training_data/feature_matrix_train.csv \
  --test-matrix  lasso_training_data/feature_matrix_test.csv \
  --alpha 0.000174 \
  --out-dir model_artifacts
```
> Run 'extract_features.py' to strip features from training matrix for Nextflow run (see docstring). 

## 🧩 API quickstart (Fargate)

> **Note on URLs (security)**
> This repo doesn’t publish a live endpoint. The ALB DNS you create in AWS is a **public, unauthenticated entry point**. Posting it in a README invites scraping/abuse, noisy logs, and surprise costs.  
> Use a placeholder like `https://<YOUR_API_URL>` here, and deploy behind one of these:
> - **API Gateway → VPC Link → private ALB** (recommended): add API keys/JWT, throttling, custom domain + TLS.
> - **Public ALB** (testing only): restrict Security Group to your IPs, enable TLS (ACM), require `X-API-Key`, attach WAF.
>
> When testing locally, substitute your own URL **without** committing it:
> ```bash
> export API_BASE="https://<YOUR_API_URL>"
> curl -H "X-API-Key: $API_KEY" "$API_BASE/health"
> ```

Replace `<ALB_DNS>` and `<YOUR_API_URL>` with your own when available.

**Predict from inline features**
```bash
curl -X POST http://<ALB_DNS>/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "sample_id":"EX123",
    "features":{"S_1434":1,"ORF1ab_2428":1,"S_645":0}
  }'
```

**Presign an S3 PUT (upload)**
```
curl -X POST http://<ALB_DNS>/presign \
  -H 'Content-Type: application/json' \
  -d '{"key":"uploads/example.fasta","method":"put_object","expires_in":3600}'
```
**Queue a batch job**
```
curl -X POST http://<ALB_DNS>/submit \
  -H 'Content-Type: application/json' \
  -d '{
    "input_s3":"s3://ach-covid-lasso-us-east-2/inputs/batch/*.fasta",
    "output_s3":"s3://ach-covid-lasso-us-east-2/results/batch-001/",
    "params":{"profile":"aws"}
  }'
```
### 🔧 Environment (ECS task)

**Required**
```
AWS_REGION=us-east-2
S3_BUCKET=<your-s3-bucket>                  # e.g., covid-cfr-prod-us-east-2
SQS_QUEUE_URL=https://sqs.<region>.amazonaws.com/<account-id>/<queue-name>
```

**Optional**
```
REQUIRE_API_KEY=true                        # set true to enforce API key auth
API_KEY=<strong-secret-value>               # store in Secrets Manager/Task secrets
MODEL_VERSION=v1
```

---

## 🧬 Workflow Overview
```mermaid
flowchart LR
  A["Fetch genomes"] --> B["Reference genome"]
  B --> C["MAFFT alignment"]
  C --> D["Filter by % identity"]
  D --> E["Binary variant matrix"]
  E --> F["Lasso regression"]
  F --> G["Explain (SHAP/LIME)"]
  D --> H["DNABERT fine-tuning (trained; NF integration pending)"]
  E --> I["Visualizations"]
  F --> J["Collapse features"]
  H --> K["Predictions"]
  J --> K
```

---

## ⚙️ Productionization

- **CI/CD**: GitHub Actions builds `docker/Dockerfile.lasso` and pushes to ECR (`latest` + `<commit_sha>`).
- **Batch scoring**: Run large cohorts on AWS Batch.
- **On-demand inference**: FastAPI microservice on ECS Fargate, exposed via API Gateway (planned).
- **Data exchange**: Pre-signed S3 URLs for secure input/output (planned).
- **Async workflows**: SQS for job orchestration (planned).
- **Observability**: **Amazon CloudWatch** Logs & Metrics with Alarms (job failures, p95 latency, SQS queue depth).
- **Monitoring**: Drift reports for variant distribution shifts (planned).

### CI/CD: GitHub Actions → Docker → AWS ECR (Shipped)

This repo auto-builds a Docker image and pushes it to **Amazon ECR** on every push to `main`.

- **Workflow:** `.github/workflows/ecr-push.yml`
- **Dockerfile:** `docker/Dockerfile.lasso`
- **Registry:** `802861900950.dkr.ecr.us-east-2.amazonaws.com/covid-lasso`
- **Tags pushed:** `:latest` (convenience) and `:<commit_sha>` (immutable)
- **Build cache:** an extra `:buildcache` (or `:cache`) tag is used by Buildx for faster builds.

Pull the image:
```bash
Registry: <account-id>.dkr.ecr.<region>.amazonaws.com/<repo>
...
docker pull <account-id>.dkr.ecr.<region>.amazonaws.com/<repo>:latest
docker pull <account-id>.dkr.ecr.<region>.amazonaws.com/<repo>:<commit_sha>
```

### 🛰️ Deployed API (ECS Fargate + ALB) (Shipped)
Public ALB → FastAPI service on ECS Fargate. Health:
_Currently private_
```bash
curl -H "X-API-Key: $API_KEY" "$API_BASE/health"
# {"status":"ok","time":...}
```

### 🔒 Security 
Endpoints can require an API key (`REQUIRE_API_KEY=true`). S3 access uses presigned URLs scoped to allowed prefixes; SQS messages are JSON-validated server side. CI uses AWS OIDC (no long-lived keys).

---



## 💻 CLI by Stage

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

### 3) Train & evaluate Lasso
```bash
(cd scripts && python ML_model.py \
    --train-matrix ../lasso_training_data/feature_matrix_train.csv \
    --test-matrix  ../lasso_training_data/feature_matrix_test.csv \
    --alpha 0.000174 \
    --out-dir ../model_artifacts)
```

### 4) Robustness: negative controls & ablations
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

### 5) Model explanations (SHAP & LIME)
```bash
(cd scripts && python explain_lasso.py \
    --train_csv ../lasso_training_data/feature_matrix_train.csv \
    --test_csv  ../lasso_training_data/feature_matrix_test.csv  \
    --artifacts_dir ../model_artifacts \
    --outdir ../explanations \
    --lime_n 5 --lime_select largest_error --lime_space raw --lime_digits 6)
```

### 6) Collapse and predict on new genomes
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
```bash
# Heatmap
(cd scripts && python Variant_feature_heatmap.py)
# Regularization curve
(cd scripts && python Regularization.py)
```

### 8) DNABERT baseline (trained; NF integration pending)
Fine-tuned DNABERT regressor has been trained separately; Nextflow module will be added in v1.1.

---

## 🧭 Nextflow Entrypoint
A Nextflow wrapper (`main.nf`) orchestrates the stages above for scalable, parallel execution.

**Typical usage**
```bash
# Local execution
nextflow run main.nf -profile local \
  --samples "transformed_data/variant_samples_small.fasta" \
  --outdir results_nf

# Docker execution (per-process containers)
nextflow run main.nf -profile docker \
  --samples "transformed_data/variant_samples_small.fasta" \
  --outdir results_nf

# AWS Batch execution (cloud computing)
nextflow run main.nf -profile aws \
--samples "s3://<your-bucket>/inputs/test_samples/*.fasta"
--outdir "s3://<your-bucket>/results_test"
```

See `nextflow.config` for available profiles (e.g., `local`, `docker`), full list of CLI args and tunables like CPUs/memory, container images, and work directory. Override at runtime with `-with-report`, `-with-trace`, `-with-dag flowchart.png`, and resume with `-resume`.

---

## 📊 Outputs
- **Model artifacts (Lasso):** `model_artifacts/lasso_model.joblib`, `model_artifacts/scaler.joblib`
- **Model artifacts (DNABERT — trained; NF integration pending):**
  - Weights: `dnabert_cfr_regressor_all_layers_richer_head/` (Keras v3 SavedModel), `model_artifacts/best_model.h5` (HDF5 snapshot)
  - Tokenizer also `dnabert_cfr_regressor_all_layers_richer_head/` containing:
    - `tokenizer_config.json` *(BertTokenizer; model_max_length=512; do_lower_case=false)*
    - `special_tokens_map.json` *([CLS], [SEP], [MASK], [PAD], [UNK])*
    - `vocab.txt` *(4,101 tokens; 6-mers + specials)*
- **Explanations:** SHAP plots, LIME HTMLs, feature importance tables
- **QC reports:** Identity thresholds, rejected sequences
- **Visualizations:** Variant heatmaps, regularization curves
- **Predictions:** Collapsed feature matrices + CFR predictions

---

## 📌 Roadmap

### ✅ Shipped — v1.1 (Sep 2025)
- Nextflow DSL2 orchestration
- Dockerized stages with pinned dependencies
- **AWS Batch** integration for distributed, chunked scoring
- **FastAPI microservice on ECS Fargate (ALB)** — `/health`, `/predict` (stub), `/presign` (S3 presigned URLs), `/submit` (enqueue to SQS)
- CI/CD: GitHub Actions (OIDC) → Docker Buildx → **Amazon ECR** (immutable `:<commit_sha>` + `:latest`)
- Lasso baseline with bootstrap validation
- Robustness suite (label permutations, feature shuffles, ablations)
- Explainability (SHAP, LIME)
- Visualizations & reports (heatmaps, regularization curves)
- Reproducibility artifacts (models, scalers, run reports)

### 🚧 In Flight — v1.2
- **API Gateway + auth/rate limits** in front of Fargate (private ALB via VPC Link)
- MLflow experiment tracking + model registry (design complete)
- Wire real model artifacts into `/predict` (scaler + Lasso), improve schema validation
- **Integrate trained DNABERT module into Nextflow** (optional GPU Batch queue)

### 🔮 Planned Enhancements — v2 (optional)
- **Presigned S3 I/O + SQS**: end-to-end async batch (worker consumer + status)
- Drift monitoring job & report (feature/label distribution shifts)
- Streamlit/Gradio dashboard for explanations
- Multi-omics extension (RNA-seq, host genetics)

---

## 🧰 Troubleshooting
- **CloudWatch Logs Insights**: grep structured JSON logs to find failing chunks quickly. Useful starter query:
```sql
fields @timestamp, @message
| filter level = 'ERROR'
| sort @timestamp desc
| limit 50
```
- **MAFFT not found**: ensure it is installed and on `PATH`.
- **Path errors**: some scripts assume execution from `scripts/`; either run from there or adjust relative paths.
- **Nextflow var errors**: confirm `main.nf` input channels match declared parameters; use `-with-dag` to inspect graph.
- **CSV parsing**: prefer `awk`/`csvkit` for large files; ensure correct delimiters (`,` for CSV, `\t` for TSV).
- **I/O-bound runs**: consider enabling Fusion S3 streaming or using local ephemeral storage for heavy intermediates.

---

## 📜 License
Apache-2.0.
