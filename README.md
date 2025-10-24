# DietNet_dual_plus
**DietNet-dual-plus** is a dual-input deep learning model developed to predict chemotherapy response in pancreatic cancer using **pre-treatment FNA specimens** and optional clinicopathologic features.  
This model extends the original open-source [DietNet](https://github.com/ljstrnadiii/DietNet) architecture by adding:
- Dual-input integration (genomic + clinicopathologic branches)
- L1 regularization and dropout optimization
- Repeated cross-validation and bootstrap validation
- SHAP-based interpretability
- Δ(FNA→Tumor) and tumor GSVA association analyses

---

##  Installation

Python 3.10+ is recommended.

Usage:
  python dietnet_dual_plus.py --csv main.csv --outdir out/ [options]
##  Input Format

**Main CSV file:**
Each row represents a sample.

| Column | Description |
|--------|--------------|
| `sample` | Sample ID (e.g., NAT_001_FNA or NAT_001_tumor) |
| `CAP_binary` | Response label: 1 = Responder (CAP 1/2), 0 = Non-responder (CAP 3) |
| `Genomic features` | Binary (0/1) features representing mutation, CNV, or LOH |
| `Clinical features` | Optional numeric or categorical variables |

**Optional inputs:**
- `--covars_csv`: Clinical covariates file (merged via `sample` column)
- `--gsva_tumor_tsv`: Tumor GSVA pathway scores
- `--external_csv`: External cohort for inference

---

##  Usage

### 1. Training with 5-fold Cross-Validation
```bash
python src/dietnet_dual_plus.py   --csv main.csv --outdir out/   --kfold 5 --repeats 200 --hidden 64,32 --epochs 200 --lr 1e-3 --w_l1 1e-4   --opt_threshold
```

### 2. Adding Clinical Covariates
```bash
python src/dietnet_dual_plus.py   --csv main.csv --covars_csv covars.csv   --covar_cols Age,Sex,Morphology   --covar_select_l1 --covar_select_maxk 10   --outdir out_cov/
```

### 3. Platform Adjustment
```bash
python src/dietnet_dual_plus.py   --csv main.csv --platform_col seq_method --platform_adjust stratify
```

### 4. Paired Δ(FNA→Tumor) and GSVA Analyses
```bash
python src/dietnet_dual_plus.py   --csv main.csv --gsva_tumor_tsv tumor_gsva.tsv --outdir out_mech/
```

### 5. External Cohort Inference
```bash
python src/dietnet_dual_plus.py   --csv main.csv --external_csv external.csv --outdir out_ext/
```

---

## 📊 Outputs

| File | Description |
|------|--------------|
| `fna_cv_summary.txt` | Cross-validation metrics |
| `fna_top_features.csv` | Ranked feature importance |
| `fna_feature_stability.csv` | Bootstrap feature stability |
| `delta_univariate.csv` | Paired Δ(FNA→Tumor) results |
| `tumor_gsva_univariate.csv` | Tumor GSVA pathway associations |
| `external_predictions.csv` | Predictions on external dataset |

---

##  Model Overview

- **Input:** Genomic and clinicopathologic feature matrices  
- **Architecture:** Dual-branch feedforward MLP with feature-wise L1 regularization  
- **Output:** Sigmoid probability of chemotherapy response  
- **Validation:** 5-fold patient-level cross-validation (200 repeats)  
- **Explainability:** SHAP feature importance and decision plots

  
##  Origin and Acknowledgment
This repository is distributed under the **MIT License**.  
Based in part on the open-source [DietNet](https://github.com/ljstrnadiii/DietNet) implementation by Ljstrnad et al.

We gratefully acknowledge the original authors for releasing the DietNet codebase.
