# DietNet_dual_plus
**DietNet-dual-plus** is a dual-input deep learning model developed to predict chemotherapy response in pancreatic cancer using **pre-treatment FNA specimens** and optional clinicopathologic features.
This model extends the original open-source [DietNet](https://github.com/ljstrnadiii/DietNet) architecture by adding:
- Dual-input integration (genomic + clinicopathologic branches)
- Unified imputation pipeline (`impute_and_filter` / `impute_apply`) to prevent train-test data leakage
- L1 regularization, dropout optimization, and fold-internal covariate selection
- Repeated cross-validation and bootstrap validation
- OncoKB-aware directional CNV filtering
- Platform-stratified cross-validation (e.g., WES vs targeted panel)
- SHAP-based interpretability

---

## Installation

Python 3.9+ is recommended (tested on 3.9.15 and 3.10+).

```bash
pip install -r requirements.txt
```

## Input Format

**Main CSV file:**
Each row represents a sample.

| Column | Description |
|--------|--------------|
| `sample` | Sample ID (e.g., NAT_001_FNA or NAT_001_tumor) |
| `CAP_binary` | Response label: **0 = Responder** (CAP grade 1/2), **1 = Non-responder** (CAP grade 3) |
| `Genomic features` | Binary (0/1) features representing mutation, CNV, or LOH |
| `Clinical features` | Optional numeric or categorical variables |

**Optional inputs:**
- `--covars_csv`: Clinical covariates file (merged via `sample` column)
- `--gsva_tumor_tsv`: Tumor GSVA pathway scores
- `--external_csv`: External cohort for inference
- `--oncokb_path`: OncoKB cancer gene list for directional CNV filtering
- `--pathway_csv`: Pathway annotation for gene-pathway mapping

---

## Usage

### 1. Training with 5-fold Cross-Validation
```bash
python dietnet_dual_plus.py \
  --csv main.csv --outdir out/ \
  --kfold 5 --repeats 200 --hidden 64,32 --epochs 200 --lr 1e-3 --w_l1 1e-4 \
  --opt_threshold
```

### 2. Adding Clinical Covariates with L1 Selection
```bash
python dietnet_dual_plus.py \
  --csv main.csv --covars_csv covars.csv \
  --covar_cols Age,Sex,Morphology \
  --covar_select_l1 --covar_select_maxk 10 \
  --outdir out_cov/
```

### 3. Platform-Stratified Cross-Validation
```bash
python dietnet_dual_plus.py \
  --csv main.csv --platform_col seq_method --platform_adjust stratify \
  --outdir out_platform/
```

### 4. OncoKB Directional CNV Filtering
```bash
python dietnet_dual_plus.py \
  --csv main.csv --oncokb_path cancerGeneList.tsv \
  --collapse_gene_abn --restrict_to_pcga \
  --outdir out_oncokb/
```

### 5. Paired Delta(FNA to Tumor) and GSVA Analyses
```bash
python dietnet_dual_plus.py \
  --csv main.csv --gsva_tumor_tsv tumor_gsva.tsv --outdir out_mech/
```

### 6. External Cohort Inference
```bash
python dietnet_dual_plus.py \
  --csv main.csv --external_csv external.csv --outdir out_ext/
```

---

## Outputs

| File | Description |
|------|--------------|
| `fna_cv_summary.txt` | Cross-validation metrics (pooled OOF AUC, fold-mean AUC, bootstrap-corrected AUC) |
| `fna_cv_repeats_summary.csv` | Per-repeat AUC summary |
| `fna_top_features.csv` | Ranked feature importance |
| `fna_feature_stability.csv` | Bootstrap feature stability |
| `delta_univariate.csv` | Paired Delta(FNA to Tumor) results |
| `tumor_gsva_univariate.csv` | Tumor GSVA pathway associations |
| `external_predictions.csv` | Predictions on external dataset |
| `external_metrics.txt` | External validation AUC metrics |

---

## Model Overview

- **Input:** Genomic and clinicopathologic feature matrices
- **Architecture:** Dual-branch feedforward MLP with feature-wise L1 regularization
- **Output:** Sigmoid probability of non-response (P(Non-responder))
- **Validation:** 5-fold patient-level cross-validation (200 repeats)
- **Imputation:** Unified `impute_and_filter()` / `impute_apply()` pipeline ensures no train-test leakage
- **Explainability:** SHAP feature importance and decision plots

## Version History

- **v4.0** (2026-03-06): Unified imputation pipeline, fold-internal covariate selection, OncoKB directional CNV filtering, platform stratification, NaN-safe handling
- **v3.0** (2026-03-04): Honest CV (no test-fold leak in early stopping)
- **v2.0** (2026-03-03): Death column encoding fix, HRD removal
- **v1.0** (2025-10): Initial release

## Origin and Acknowledgment
This repository is distributed under the **MIT License**.
Based in part on the open-source [DietNet](https://github.com/ljstrnadiii/DietNet) implementation by Ljstrnad et al.

We gratefully acknowledge the original authors for releasing the DietNet codebase.
