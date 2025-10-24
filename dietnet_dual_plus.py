#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dietnet_dual_plus.py
--------------------
One-command pipeline that does BOTH:
  (1) Pre-treatment prediction (clinical use): FNA-only DietNet-style model with stratified *grouped* K-fold CV
      + optional clinical covariates as features
      + optional external cohort application (train on all FNA, predict external FNA)
  (2) Mechanism/association (interpretation): Delta features (tumor - FNA) with patient-level tests,
      and optional tumor GSVA (continuous) associations (univariate & multivariate).

Inputs
- Main wide CSV: rows = samples; columns include:
    * 'sample' (e.g., NAT123_FNA / NAT123_tumor)
    * binary label (default: CAP_binary 0/1)
    * genomic features: 0/1/NaN (mutation/CNV/LOH etc.); NaN->0 for modeling
    * optional meta columns (drop via --drop_cols)
- Optional covariates CSV (joined by 'sample'): columns to include as features via --covar_cols
- Optional tumor GSVA TSV: row index or first column is 'sample', remaining columns are pathways (continuous)

Outputs (to --outdir):
  - fna_cv_summary.txt, fna_top_features.csv
  - delta_univariate.csv, delta_multivariate_top.csv
  - tumor_gsva_univariate.csv (if GSVA provided)
  - tumor_gsva_multivariate_top.csv (if GSVA provided)
  - external_predictions.csv, external_metrics.txt (if external CSV provided)
"""

import os, sys, argparse

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

# === PCGA v3 gene set (hard-coded as requested) ===
PCGA_V3_SET = set([
    'ARID1A','NRAS','AKT3','ERBB4','TGFBR2','CTNNB1','PBRM1','PIK3CA',
    'APC','BRAF','FGFR1','SMARCA2','CDKN2A','TGFBR1','PTEN','HRAS',
    'ATM','KRAS','BRCA2','AKT1','AXIN1','PALB2','TP53','MAP2K4',
    'NF1','ERBB2','BRCA1','RNF43','GATA6','SMAD4','STK11','SMARCA4',
    'AKT2','GNAS','KDM6A','RBM10'
])

def _feature_to_gene(feat_name: str) -> str:
    """Heuristic: map feature column name to gene symbol by taking the token before the first '_' or '.'.
    Examples: 'KRAS_mut' -> 'KRAS', 'TP53.del' -> 'TP53', 'PIK3CA_amp_cov' -> 'PIK3CA'"""
    s = str(feat_name)
    s = s.split('_', 1)[0]
    s = s.split('.', 1)[0]
    return s.upper()

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Torch (optional)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_OK = True
except Exception:
    TORCH_OK = False

# CV splitters
from sklearn.model_selection import StratifiedKFold, GroupKFold
try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_SGKF = True
except Exception:
    HAS_SGKF = False

from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, roc_curve
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.utils import resample
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from numpy import interp

try:
    from scipy.stats import fisher_exact, mannwhitneyu, ttest_ind
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


# Preferred device selector (honors user preference; otherwise MPS > CUDA > CPU)
def pick_device(prefer: str = "auto"):
    if not TORCH_OK:
        raise RuntimeError("PyTorch is not available. Please install PyTorch to use this functionality.")
    prefer = (prefer or "auto").lower()
    if prefer in ("cpu", "cuda", "mps"):
        try:
            if prefer == "cuda" and torch.cuda.is_available():
                return torch.device("cuda")
            if prefer == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            if prefer == "cpu":
                return torch.device("cpu")
        except Exception:
            pass
        # fallback to auto if preferred is not actually available
    # auto: try mps -> cuda -> cpu
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            return torch.device("cuda")
    except Exception:
        pass
    return torch.device("cpu")


# ---------------------------
# Utilities
# ---------------------------

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_OK:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def base_id(sample_name: str) -> str:
    s = str(sample_name)
    if s.endswith("_FNA"):
        return s[:-4]
    if s.endswith("_tumor"):
        return s[:-6]
    return s

def zscore_rows(mat: np.ndarray) -> np.ndarray:
    m = mat.mean(axis=1, keepdims=True)
    s = mat.std(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return (mat - m) / s

def parse_hidden(hidden_str: str) -> List[int]:
    return [int(x) for x in hidden_str.split(",") if x.strip()]
    
def logits_to_probs(logits: np.ndarray) -> np.ndarray:
    """Convert logits to probabilities using softmax for class 1."""
    exps = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exps[:, 1] / exps.sum(axis=1)

def youden_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Return threshold that maximizes (TPR - FPR). Robust to NaN/Inf and degenerate splits."""
    y = np.asarray(y_true)
    p = np.asarray(probs, dtype=float)
    # sanitize probabilities
    if not np.all(np.isfinite(p)):
        mean_p = np.nanmean(p)
        if not np.isfinite(mean_p):
            mean_p = 0.5
        p = np.nan_to_num(p, nan=mean_p, posinf=0.5, neginf=0.5)
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask].astype(int)
    p = p[mask]
    if y.size == 0 or len(np.unique(y)) < 2:
        return 0.5
    try:
        fpr, tpr, thr = roc_curve(y, p)
    except Exception:
        return 0.5
    j = tpr - fpr
    return float(thr[np.nanargmax(j)])

def metrics_from_probs(probs: np.ndarray, y_true: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    y_pred = (probs >= thr).astype(int)
    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, probs) if len(np.unique(y_true)) == 2 else np.nan
    except ValueError:
        auc = np.nan
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TN, FP, FN, TP = cm.ravel()
    sens = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    spec = TN / (TN + FP) if (TN + FP) > 0 else np.nan
    return dict(auc=auc, acc=acc, sens=sens, spec=spec, cm=cm)

def prune_features(df_feat: pd.DataFrame, min_prevalence: float = 0.0, max_corr: float = 1.01) -> pd.DataFrame:
    """
    - Drop features with prevalence < min_prevalence (for binary-like 0/1 columns)
    - Drop one of each pair with |corr| > max_corr (e.g., 0.95)
    """
    X = df_feat.copy()
    # prevalence pruning (for ~binary columns; keep mid-prevalence)
    if min_prevalence > 0:
        prev = X.mean(axis=0, skipna=True)
        keep = prev.index[(prev >= min_prevalence) & (prev <= (1 - min_prevalence))]
        X = X.reindex(columns=keep)
    # correlation pruning
    if max_corr < 1.0 and X.shape[1] > 1:
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        drop_cols = [c for c in upper.columns if (upper[c] > max_corr).any()]
        X = X.drop(columns=drop_cols, errors="ignore")
    return X
def metrics_from_logits(logits: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    """Compute metrics from logits by converting to probabilities first."""
    # logits: (N,2)
    probs = logits_to_probs(logits)
    y_pred = (probs >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, probs) if len(np.unique(y_true)) == 2 else np.nan
    except ValueError:
        auc = np.nan
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    TN, FP, FN, TP = cm.ravel()
    sens = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    spec = TN / (TN + FP) if (TN + FP) > 0 else np.nan
    return dict(auc=auc, acc=acc, sens=sens, spec=spec, cm=cm)


# ---------------------------
# Plotting & analysis helpers (ROC, CI, DCA, CV collection)
# ---------------------------

def collect_cv_predictions(X, y, kfold, hidden, epochs, lr, w_l1, seed, groups=None, y_strat=None, device=None, time_profile: bool = False, use_batchnorm: bool = False):
    """Run CV and return per-fold (y_true, y_prob), plus fold-wise AUCs."""
    y_for_split = y if y_strat is None else y_strat
    if groups is not None:
        if HAS_SGKF:
            splitter = StratifiedGroupKFold(n_splits=kfold, shuffle=True, random_state=seed)
            splits = splitter.split(X, y_for_split, groups)
        else:
            print("[WARN] StratifiedGroupKFold unavailable; falling back to GroupKFold (no stratification).")
            splitter = GroupKFold(n_splits=kfold)
            splits = splitter.split(X, groups=groups)
    else:
        splitter = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=seed)
        splits = splitter.split(X, y_for_split)

    fold_preds = []
    fold_aucs = []
    for fold_idx, (tr, te) in enumerate(splits, start=1):
        # Validate non-empty splits
        if len(tr) == 0 or len(te) == 0:
            print(f"[WARN] Fold {fold_idx}: Skipping fold with empty train ({len(tr)}) or test ({len(te)}) set")
            continue
        m, Ue, probs_te, y_te = dietnet_fold_torch(X[tr], y[tr], X[te], y[te], hidden, epochs, lr, w_l1, seed,
                                                   device=device, time_profile=time_profile, use_batchnorm=use_batchnorm)
        # Append also the test indices for each fold
        fold_preds.append((y_te.astype(int), probs_te.astype(float), te))
        try:
            from sklearn.metrics import roc_auc_score
            fold_aucs.append(roc_auc_score(y_te, probs_te))
        except Exception:
            fold_aucs.append(np.nan)
    return fold_preds, np.array(fold_aucs, dtype=float)


def plot_cv_roc_with_ci(fold_preds, out_png=None, title_prefix="ROC (CV)"):
    """Plot thin ROC per fold + mean ROC + 95% CI(AUC). Returns (auc_mean, ci_low, ci_high)."""
    # Compute per-fold ROC and interpolate to a common grid
    mean_fpr = np.linspace(0, 1, 101)
    tprs = []
    aucs = []
    valid_entries = []
    for fold_idx, item in enumerate(fold_preds, start=1):
        y_true, y_prob = item[0], item[1]
        if len(np.unique(y_true)) < 2:
            print(f"[ROC] Fold {fold_idx}: skipped (single-class y_true)")
            continue
        try:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc = roc_auc_score(y_true, y_prob)
        except ValueError as e:
            print(f"[ROC] Fold {fold_idx}: skipped ({e})")
            continue
        aucs.append(auc)
        tpr_interp = interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)
        valid_entries.append((fpr, tpr))
    if len(tprs) == 0:
        print("[ROC] No valid folds for ROC plotting.")
        if out_png:
            plt.figure(figsize=(5.5, 5.5), dpi=200)
            plt.title(f"{title_prefix}\nAUC=N/A")
            plt.xlabel("1 - Specificity")
            plt.ylabel("Sensitivity")
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.grid(True, linewidth=0.5, alpha=0.5)
            plt.savefig(out_png, bbox_inches="tight")
            plt.close()
        return np.nan, np.nan, np.nan
    tprs = np.array(tprs)
    mean_tpr = np.nanmean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    aucs = np.array(aucs, dtype=float)
    # 95% CI from fold AUCs (percentile)
    if np.isfinite(aucs).sum() >= 2:
        ci_low, ci_high = np.nanpercentile(aucs, [2.5, 97.5])
    else:
        ci_low = ci_high = np.nan
    auc_mean = np.nanmean(aucs)

    fig = plt.figure(figsize=(5.5,5.5), dpi=200)
    ax = plt.gca()
    # Thin ROC lines per fold
    for (fpr, tpr) in valid_entries:
        ax.plot(fpr, tpr, alpha=0.3, linewidth=1)
    # Mean ROC
    ax.plot([0,1],[0,1], linestyle="--")
    ax.plot(mean_fpr, mean_tpr, linewidth=2)
    subtitle = f"AUC={auc_mean:.3f} (95% CI {ci_low:.3f}-{ci_high:.3f})" if np.isfinite(auc_mean) else "AUC=N/A"
    ax.set(title=f"{title_prefix}\n{subtitle}", xlabel="1 - Specificity", ylabel="Sensitivity", xlim=(0,1), ylim=(0,1))
    ax.grid(True, linewidth=0.5, alpha=0.5)
    if out_png:
        fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return float(auc_mean) if np.isfinite(auc_mean) else np.nan, float(ci_low) if np.isfinite(ci_low) else np.nan, float(ci_high) if np.isfinite(ci_high) else np.nan


def decision_curve(y_true, y_prob, thresholds=np.linspace(0.1,0.5,41)):
    """Return thresholds, net benefit for model, treat-all, and treat-none (0)."""
    y = np.asarray(y_true).astype(int); p = np.asarray(y_prob, dtype=float)
    N = y.size
    nets = []
    treat_all = []
    for pt in thresholds:
        # Skip invalid threshold values
        if pt >= 1.0 or pt <= 0.0:
            nets.append(np.nan)
            treat_all.append(np.nan)
            continue
        pred = (p >= pt).astype(int)
        TP = ((pred==1) & (y==1)).sum()
        FP = ((pred==1) & (y==0)).sum()
        NB = TP/N - FP/N * (pt/(1-pt))
        nets.append(NB)
        # treat-all baseline
        TP_a = (y==1).sum(); FP_a = (y==0).sum()
        NB_all = TP_a/N - FP_a/N * (pt/(1-pt))
        treat_all.append(NB_all)
    return thresholds, np.array(nets, dtype=float), np.array(treat_all, dtype=float)


def plot_decision_curve(thr, nb, nb_all, out_png=None):
    fig = plt.figure(figsize=(6,4), dpi=200)
    ax = plt.gca()
    ax.plot(thr, nb, label="Model")
    ax.plot(thr, nb_all, linestyle="--", label="Treat-all")
    ax.axhline(0, linestyle=":", label="Treat-none")
    ax.set(xlabel="Threshold probability", ylabel="Net benefit", title="Decision Curve Analysis")
    ax.legend()
    ax.grid(True, linewidth=0.5, alpha=0.5)
    if out_png:
        fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# ---------------------------
# DietNet-style model (Torch)
# ---------------------------
# Based on Diet Networks: "Diet Networks: Thin Parameters for Fat Genomic Data"
# Original paper: https://openreview.net/forum?id=Sk-oDY9ge
# Reference implementation: https://github.com/ljstrnadiii/DietNet
#
# Key Innovation: Instead of learning weight matrices W directly (p×hidden_dim parameters),
# DietNet uses an auxiliary network to generate W from feature descriptors D, dramatically
# reducing parameters for high-dimensional genomic data.
#
# Architecture:
#   1. Feature descriptor D: (n_features, d_desc) matrix capturing feature relationships
#      - Original: SNP histogram embeddings (external biological knowledge)
#      - This implementation: Z-scored training data profiles (data-driven)
#   2. Auxiliary Network (feature_mlp): Learns to map D → Ue (per-feature weights)
#   3. Discriminative Network: logits = X @ Ue + bias
#
# Adaptations for clinical genomics:
#   - Data-driven descriptors instead of pre-computed histograms (more general)
#   - L1 regularization instead of reconstruction loss (better interpretability)
#   - Optional BatchNorm for stability with large-scale data
# ---------------------------

class DietNetTorch(nn.Module):
    """
    DietNet-style auxiliary network for parameter-efficient genomic prediction.

    Generates per-feature weights Ue from descriptors D using an auxiliary MLP.
    Final prediction: logits = X @ Ue + bias

    Parameters:
        d_desc: Dimension of feature descriptors (typically n_train samples)
        n_features: Number of input features (genes/mutations)
        n_classes: Number of output classes (typically 2 for binary classification)
        hidden: List of hidden layer sizes for auxiliary network
        use_batchnorm: Enable batch normalization for large-scale data stability

    Reference:
        Romero et al. "Diet Networks: Thin Parameters for Fat Genomic Data"
        ICLR 2017. https://github.com/ljstrnadiii/DietNet
    """
    def __init__(self, d_desc: int, n_features: int, n_classes: int, hidden: List[int], use_batchnorm: bool = False):
        super().__init__()
        layers = []
        prev = d_desc
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.feature_mlp = nn.Sequential(*layers)
        self.bias = nn.Parameter(torch.zeros(n_classes))
        self.Ue = None

    def forward(self, X, D):
        """
        Forward pass using DietNet architecture.

        Args:
            X: Input data (batch_size, n_features)
            D: Feature descriptors (n_features, d_desc)

        Returns:
            logits: (batch_size, n_classes)
        """
        # Auxiliary network: Generate per-feature weights from descriptors
        self.Ue = self.feature_mlp(D)              # (n_features, n_classes)
        # Discriminative network: Weighted sum of features
        logits = X @ self.Ue + self.bias           # (batch_size, n_classes)
        return logits


def dietnet_fold_torch(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, y_te: np.ndarray,
                       hidden: List[int], epochs: int, lr: float, w_l1: float, seed: int,
                       device: torch.device | None = None, time_profile: bool = False, use_batchnorm: bool = False):
    # --- Fold-wise imputation (robust, minimal) ---
    X_tr = np.asarray(X_tr, dtype=float)
    X_te = np.asarray(X_te, dtype=float)
    # Detect binary vs continuous on the TRAIN fold only
    bin_mask = []
    for j in range(X_tr.shape[1]):
        vals = X_tr[:, j]
        u = np.unique(vals[~np.isnan(vals)])
        bin_mask.append(set(u.tolist()).issubset({0.0, 1.0}))
    bin_mask = np.array(bin_mask, dtype=bool)
    cont_mask = ~bin_mask
    # Impute: binary -> 0, continuous -> train median
    X_tr_imp = X_tr.copy(); X_te_imp = X_te.copy()
    if bin_mask.any():
        tr_nan = np.isnan(X_tr_imp[:, bin_mask]); te_nan = np.isnan(X_te_imp[:, bin_mask])
        X_tr_imp[:, bin_mask][tr_nan] = 0.0
        X_te_imp[:, bin_mask][te_nan] = 0.0
    for j in np.where(cont_mask)[0].tolist():
        med = np.nanmedian(X_tr_imp[:, j])
        if not np.isfinite(med):
            med = 0.0
        X_tr_imp[:, j] = np.where(np.isnan(X_tr_imp[:, j]), med, X_tr_imp[:, j])
        X_te_imp[:, j] = np.where(np.isnan(X_te_imp[:, j]), med, X_te_imp[:, j])
    # Drop constant columns (train fold) to avoid degenerate design
    std_tr = np.nanstd(X_tr_imp, axis=0)
    keep_cols = std_tr > 0
    if keep_cols.sum() == 0:
        # keep at least one column
        if len(std_tr) > 0:
            # Keep the column with maximum std (or first if all NaN)
            idx = np.nanargmax(std_tr) if np.isfinite(std_tr).any() else 0
            keep_cols[idx] = True
        else:
            raise ValueError("Cannot proceed with zero features after constant column filtering")
    X_tr = X_tr_imp[:, keep_cols]
    X_te = X_te_imp[:, keep_cols]

    device = device or pick_device()
    import time
    t0 = time.time()
    Xtr = torch.tensor(X_tr, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_tr, dtype=torch.long, device=device)
    Xte = torch.tensor(X_te, dtype=torch.float32, device=device)
    yte = torch.tensor(y_te, dtype=torch.long, device=device)

    # Feature descriptors D: Z-scored training data profiles (data-driven)
    # Each feature is described by its pattern across all training samples
    D = torch.tensor(zscore_rows(X_tr.T), dtype=torch.float32, device=device)  # (n_features, n_train)
    d_desc = D.shape[1]
    n_features = X_tr.shape[1]

    model = DietNetTorch(d_desc=d_desc, n_features=n_features, n_classes=2, hidden=hidden, use_batchnorm=use_batchnorm).to(device)
    pos = (ytr == 1).sum().item(); neg = (ytr == 0).sum().item()
    w_pos = max(neg / max(pos, 1), 1.0)
    class_weight = torch.tensor([1.0, w_pos], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_auc, best_state = -np.inf, None
    patience = max(epochs // 5, 10); counter = 0
    for ep in range(1, epochs+1):
        model.train()
        opt.zero_grad()
        logits = model(Xtr, D)
        loss = criterion(logits, ytr)
        l1 = model.Ue.abs().sum() if model.Ue is not None else torch.tensor(0., device=device)
        loss = loss + w_l1 * l1
        loss.backward()
        opt.step()

        if time_profile and ep % 10 == 0:
            try:
                if device.type == "mps" and hasattr(torch.backends, "mps"):
                    # MPS synchronization (may not be available in all PyTorch versions)
                    if hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
                        torch.mps.synchronize()
                elif device.type == "cuda":
                    torch.cuda.synchronize()
            except Exception as e:
                print(f"[WARN] Device synchronization failed: {e}")
            print(f"[timing] fold_epoch={ep} elapsed={time.time()-t0:.2f}s")

        model.eval()
        with torch.no_grad():
            logits_te = model(Xte, D)
            logits_np = logits_te.detach().cpu().numpy()
            m = metrics_from_logits(logits_np, y_te)
            auc = m["auc"]
        if not np.isnan(auc) and auc > best_auc + 1e-4:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict({k: v for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        logits_te = model(Xte, D).detach().cpu().numpy()
    probs_te = logits_to_probs(logits_te)
    probs_te = np.nan_to_num(probs_te, nan=0.5, posinf=0.5, neginf=0.5)
    m = metrics_from_probs(probs_te, y_te)

    # Expand learned weights back to the original feature dimension (before constant-column drop)
    Ue_small = model.Ue.detach().cpu().numpy()            # shape: (sum(keep_cols), 2)
    orig_n = X_tr_imp.shape[1]                             # Number of features after imputation, before constant-column filtering
    Ue_full = np.zeros((orig_n, Ue_small.shape[1]), dtype=Ue_small.dtype)
    # Validate dimension consistency
    assert keep_cols.sum() == Ue_small.shape[0], f"Dimension mismatch in weight reconstruction: keep_cols.sum()={keep_cols.sum()}, Ue_small.shape[0]={Ue_small.shape[0]}"
    Ue_full[keep_cols, :] = Ue_small
    if time_profile:
        try:
            if device.type == "mps" and hasattr(torch.backends, "mps"):
                if hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
                    torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()
        except Exception as e:
            print(f"[WARN] Device synchronization failed: {e}")
        print(f"[timing] fold_total={time.time()-t0:.2f}s on {device}")
    return m, Ue_full, probs_te, y_te


def dietnet_cv(X, y, kfold, hidden, epochs, lr, w_l1, seed, groups=None, opt_threshold: bool = False, y_strat=None,
               device=None, time_profile: bool = False, use_batchnorm: bool = False):
    """Group-aware stratified CV if groups provided & StratifiedGroupKFold available.

    Args:
        use_batchnorm: Enable batch normalization in auxiliary network (recommended for large datasets, n_train > 1000)
    """
    y_for_split = y if y_strat is None else y_strat
    if groups is not None:
        if HAS_SGKF:
            splitter = StratifiedGroupKFold(n_splits=kfold, shuffle=True, random_state=seed)
            splits = splitter.split(X, y_for_split, groups)
        else:
            print("[WARN] StratifiedGroupKFold unavailable; falling back to GroupKFold (no stratification).")
            splitter = GroupKFold(n_splits=kfold)
            splits = splitter.split(X, groups=groups)
    else:
        splitter = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=seed)
        splits = splitter.split(X, y_for_split)

    aucs, accs, senss, specs = [], [], [], []
    feat_scores = []
    all_thr = []
    for fold, (tr, te) in enumerate(splits, start=1):
        # Validate non-empty splits
        if len(tr) == 0 or len(te) == 0:
            print(f"[WARN] Fold {fold}: Skipping fold with empty train ({len(tr)}) or test ({len(te)}) set")
            continue
        m, Ue, probs_te, y_te = dietnet_fold_torch(X[tr], y[tr], X[te], y[te], hidden, epochs, lr, w_l1, seed,
                                                   device=device, time_profile=time_profile, use_batchnorm=use_batchnorm)
        # Optionally optimize threshold on validation (test) fold via Youden
        if opt_threshold and len(np.unique(y_te)) == 2:
            thr = youden_threshold(y_te, probs_te)
            m = metrics_from_probs(probs_te, y_te, thr)
            all_thr.append(thr)
        aucs.append(m["auc"]); accs.append(m["acc"]); senss.append(m["sens"]); specs.append(m["spec"])
        feat_scores.append(np.abs(Ue).sum(axis=1))  # L1 across classes
        print(f"[DietNet][Fold {fold}] AUC={m['auc']:.3f} ACC={m['acc']:.3f} Sens={m['sens']:.3f} Spec={m['spec']:.3f}")
    feat_scores = np.vstack(feat_scores) if len(feat_scores)>0 else np.zeros((1, X.shape[1]))
    return (
        np.nanmean(aucs), np.nanstd(aucs),
        np.nanmean(accs), np.nanstd(accs),
        np.nanmean(senss), np.nanstd(senss),
        np.nanmean(specs), np.nanstd(specs),
        feat_scores.mean(axis=0),
        np.array(all_thr)
    )


def assess_calibration_cv(X, y, kfold, hidden, epochs, lr, w_l1, seed, groups=None):
    """Run CV and collect probabilities to compute Brier score and calibration curve."""
    if groups is not None:
        if HAS_SGKF:
            splitter = StratifiedGroupKFold(n_splits=kfold, shuffle=True, random_state=seed)
            splits = list(splitter.split(X, y, groups))
        else:
            print("[WARN] StratifiedGroupKFold unavailable; falling back to GroupKFold (no stratification).")
            splitter = GroupKFold(n_splits=kfold)
            splits = list(splitter.split(X, groups=groups))
    else:
        splitter = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=seed)
        splits = list(splitter.split(X, y))

    all_probs, all_true = [], []
    for (tr, te) in splits:
        m, Ue, probs_te, y_te = dietnet_fold_torch(X[tr], y[tr], X[te], y[te], hidden, epochs, lr, w_l1, seed)
        all_probs.append(probs_te)
        all_true.append(y_te)
    probs = np.concatenate(all_probs)
    y_true = np.concatenate(all_true)
    # Brier score
    try:
        brier = brier_score_loss(y_true, probs)
    except Exception:
        brier = np.nan
    # Calibration curve (10 bins)
    frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=10, strategy='uniform')
    calib_df = pd.DataFrame({'mean_pred': mean_pred, 'fraction_of_positives': frac_pos})
    return brier, calib_df


def permutation_test_auc(X, y, kfold, hidden, epochs, lr, w_l1, seed, groups=None, n_perm: int = 200,
                         device=None, time_profile: bool = False):
    """Label permutation test: returns list of AUC under the null."""
    rng = np.random.RandomState(seed)
    auc_null = []
    for i in range(n_perm):
        y_perm = y.copy()
        rng.shuffle(y_perm)
        (auc_m, auc_s, *_rest) = dietnet_cv(X, y_perm, kfold, hidden, epochs, lr, w_l1, seed + i, groups, opt_threshold=False,
                                            device=device, time_profile=time_profile)
        auc_null.append(auc_m)
    return np.array(auc_null)


def hparam_grid_search(X, y, groups, kfold, seed, grid_hidden, grid_wl1, grid_epochs, lr,
                      device=None, time_profile: bool = False):
    """Light-weight grid over hidden / w_l1 / epochs; returns DataFrame of results."""
    rows = []
    for H in grid_hidden:
        hidden = [int(x) for x in str(H).split(',') if str(x).strip()]
        for wl1 in grid_wl1:
            for ep in grid_epochs:
                auc_m, auc_s, acc_m, acc_s, sen_m, sen_s, spe_m, spe_s, *_ = dietnet_cv(
                    X=X, y=y, kfold=kfold, hidden=hidden, epochs=ep, lr=lr, w_l1=wl1, seed=seed,
                    groups=groups, opt_threshold=False, device=device, time_profile=time_profile)
                rows.append(dict(hidden=str(H), w_l1=wl1, epochs=ep,
                                 auc_mean=auc_m, auc_sd=auc_s,
                                 acc_mean=acc_m, acc_sd=acc_s,
                                 sens_mean=sen_m, sens_sd=sen_s,
                                 spec_mean=spe_m, spec_sd=spe_s))
    return pd.DataFrame(rows).sort_values('auc_mean', ascending=False)


def stability_selection_logit(X: pd.DataFrame, y: pd.Series, B: int = 200, subsample: float = 0.7, Cs=(1.0,), seed: int = 42) -> pd.DataFrame:
    """Stability selection with L1-logistic on subsamples. Returns selection frequency per feature."""
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    feats = X.columns.tolist()
    counts = np.zeros(len(feats), dtype=float)
    for b in range(B):
        idx = rng.choice(n, size=int(max(2, subsample * n)), replace=False)
        Xb = X.iloc[idx]
        yb = y.iloc[idx]
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("sc", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(penalty="l1", solver="saga", max_iter=3000, C=1.0, class_weight="balanced"))
        ])
        best_sel = None; best_k = -1
        for C in Cs:
            pipe.named_steps['clf'].set_params(C=C)
            pipe.fit(Xb.values, yb.values.astype(int))
            coefs = pipe.named_steps['clf'].coef_.ravel()
            sel = (coefs != 0)
            k = sel.sum()
            if k > best_k:
                best_k = k
                best_sel = sel
        if best_sel is not None:
            counts[best_sel] += 1.0
    freq = counts / float(B)
    return pd.DataFrame({"feature": feats, "stability_freq": freq}).sort_values("stability_freq", ascending=False)
def dietnet_fit_full(X: np.ndarray, y: np.ndarray, hidden: List[int], epochs: int, lr: float, w_l1: float, seed: int,
                     device=None, use_batchnorm: bool = False):
    """Fit on ALL FNA data and return a callable (predict_proba) and learned feature weights."""
    device = device or pick_device()
    Xtr = torch.tensor(X, dtype=torch.float32, device=device)
    ytr = torch.tensor(y, dtype=torch.long, device=device)

    D = torch.tensor(zscore_rows(X.T), dtype=torch.float32, device=device)
    d_desc = D.shape[1]; n_features = X.shape[1]
    model = DietNetTorch(d_desc=d_desc, n_features=n_features, n_classes=2, hidden=hidden, use_batchnorm=use_batchnorm).to(device)
    pos = (ytr == 1).sum().item(); neg = (ytr == 0).sum().item()
    w_pos = max(neg / max(pos, 1), 1.0)
    class_weight = torch.tensor([1.0, w_pos], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(1, epochs+1):
        model.train(); opt.zero_grad()
        logits = model(Xtr, D)
        loss = criterion(logits, ytr) + w_l1 * (model.Ue.abs().sum() if model.Ue is not None else 0.0)
        loss.backward(); opt.step()

    model.eval()
    def predict_proba(Xnew: np.ndarray) -> np.ndarray:
        Xte = torch.tensor(Xnew, dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = model(Xte, D).detach().cpu().numpy()
        exps = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exps[:,1] / exps.sum(axis=1, keepdims=True)[:,0]
        return probs
    return predict_proba, model.Ue.detach().cpu().numpy()


# Fit on ALL data and return (trained_model, descriptor_D_tensor), for SHAP use
def dietnet_fit_full_model(X: np.ndarray, y: np.ndarray, hidden: List[int], epochs: int, lr: float, w_l1: float, seed: int,
                           device=None, use_batchnorm: bool = False):
    """Fit on ALL data and return (trained_model, descriptor_D_tensor)."""
    device = device or pick_device()
    Xtr = torch.tensor(X, dtype=torch.float32, device=device)
    ytr = torch.tensor(y, dtype=torch.long, device=device)
    D = torch.tensor(zscore_rows(X.T), dtype=torch.float32, device=device)
    d_desc = D.shape[1]; n_features = X.shape[1]
    model = DietNetTorch(d_desc=d_desc, n_features=n_features, n_classes=2, hidden=hidden, use_batchnorm=use_batchnorm).to(device)
    pos = (ytr == 1).sum().item(); neg = (ytr == 0).sum().item()
    w_pos = max(neg / max(pos, 1), 1.0)
    class_weight = torch.tensor([1.0, w_pos], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(1, epochs+1):
        model.train(); opt.zero_grad()
        logits = model(Xtr, D)
        loss = criterion(logits, ytr) + w_l1 * (model.Ue.abs().sum() if model.Ue is not None else 0.0)
        loss.backward(); opt.step()
    model.eval()
    return model, D

class DietNetWrapper(nn.Module):
    """Wrap DietNet to fix D and accept only X for SHAP DeepExplainer."""
    def __init__(self, model: nn.Module, D: torch.Tensor):
        super().__init__()
        self.model = model
        self.D = D
    def forward(self, X):
        return self.model(X, self.D)


def make_shap_plots_torch(model: nn.Module, D: torch.Tensor, X: np.ndarray, feature_names: List[str], outdir: str, background_size: int = 100, sample_force_index: int = 0):
    """Generate SHAP summary/beeswarm and a force plot for one representative sample.
    Saves: shap_summary.png, shap_beeswarm.png, shap_force.html
    """
    try:
        import shap
    except Exception as e:
        print("[SHAP] shap is not installed; skip SHAP plots.")
        return
    os.makedirs(outdir, exist_ok=True)
    device = next(model.parameters()).device
    X_np = np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0)
    # Background (kmeans over X or random subset)
    bg_idx = np.random.RandomState(42).choice(X_np.shape[0], size=min(background_size, X_np.shape[0]), replace=False)
    background = torch.tensor(X_np[bg_idx], dtype=torch.float32, device=device)
    wrapper = DietNetWrapper(model, D)
    wrapper.eval()
    # Try DeepExplainer first; if it fails (e.g., due to SHAP version), fallback to GradientExplainer
    try:
        explainer = shap.DeepExplainer(wrapper, background)
        X_tensor = torch.tensor(X_np, dtype=torch.float32, device=device)
        # NOTE: DO NOT disable gradients here; DeepExplainer needs autograd
        shap_values = explainer.shap_values(X_tensor)
    except Exception as e:
        print(f"[SHAP] DeepExplainer failed: {e} -> fallback to GradientExplainer")
        explainer = shap.GradientExplainer(wrapper, background)
        shap_values = explainer.shap_values(X_np)

    # shap_values may be a list per class; prefer class 1 if available
    if isinstance(shap_values, list):
        sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        sv = shap_values

    # Convert to numpy if tensors
    try:
        import torch as _torch
        if isinstance(sv, _torch.Tensor):
            sv = sv.detach().cpu().numpy()
    except Exception:
        pass

    # Normalize shape:
    #  - Expect (n_samples, n_features)
    #  - If (n_samples, n_features, n_outputs) -> take class 1 (or 0)
    #  - If (n_features,) -> expand to (1, n_features)
    sv = np.asarray(sv)
    if sv.ndim == 3:
        # (N, F, C)
        C = sv.shape[-1]
        sv = sv[..., 1] if C > 1 else sv[..., 0]
    elif sv.ndim == 1:
        sv = sv.reshape(1, -1)
    # Now sv is (n_samples, n_features)
    # Summary (bar) and beeswarm
    shap.summary_plot(sv, X_np, feature_names=feature_names, plot_type="bar", show=False)
    plt.gcf().set_size_inches(7,5); plt.gcf().set_dpi(200); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "shap_summary.png"), bbox_inches="tight", dpi=300)
    plt.savefig(os.path.join(outdir, "shap_summary.pdf"), bbox_inches="tight")
    plt.close()

    shap.summary_plot(sv, X_np, feature_names=feature_names, show=False)
    plt.gcf().set_size_inches(7,5); plt.gcf().set_dpi(200); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "shap_beeswarm.png"), bbox_inches="tight", dpi=300)
    plt.savefig(os.path.join(outdir, "shap_beeswarm.pdf"), bbox_inches="tight")
    plt.close()

    # Force plot for one sample → HTML (new API; build Explanation to be robust)
    i = int(np.clip(sample_force_index, 0, X_np.shape[0]-1))
    try:
        ev = explainer.expected_value
        if isinstance(ev, (list, tuple, np.ndarray)):
            ev = ev[1] if len(np.atleast_1d(ev)) > 1 else np.atleast_1d(ev)[0]
        try:
            import torch as _torch
            if isinstance(ev, _torch.Tensor):
                ev = float(ev.detach().cpu().numpy())
        except Exception:
            pass
        # Build Explanation object (values=SHAP for class 1, base_values=ev, data=features)
        ex = shap.Explanation(values=sv[i], base_values=ev, data=X_np[i], feature_names=feature_names)
        fp = shap.plots.force(ex, matplotlib=False)
        shap.save_html(os.path.join(outdir, "shap_force.html"), fp)
    except Exception as e:
        print(f"[SHAP] force plot failed: {e}")

    # Also try a static (matplotlib) force plot for export; fallback to waterfall if needed
    try:
        plt.figure()
        ev2 = explainer.expected_value
        if isinstance(ev2, (list, tuple, np.ndarray)):
            ev2 = ev2[1] if len(np.atleast_1d(ev2)) > 1 else np.atleast_1d(ev2)[0]
        try:
            import torch as _torch
            if isinstance(ev2, _torch.Tensor):
                ev2 = float(ev2.detach().cpu().numpy())
        except Exception:
            pass
        ex2 = shap.Explanation(values=sv[i], base_values=ev2, data=X_np[i], feature_names=feature_names)
        shap.plots.force(ex2, matplotlib=True, show=False)
        plt.gcf().set_size_inches(8,3); plt.gcf().set_dpi(200); plt.tight_layout()
        plt.savefig(os.path.join(outdir, "shap_force.png"), bbox_inches="tight", dpi=300)
        plt.savefig(os.path.join(outdir, "shap_force.pdf"), bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"[SHAP] matplotlib force plot failed: {e}")
        # Fallback: waterfall plot (static)
        try:
            plt.figure()
            # Build single-sample Explanation explicitly
            ex_wf = shap.Explanation(values=sv[i], base_values=ev2, data=X_np[i], feature_names=feature_names)
            shap.plots.waterfall(ex_wf, max_display=20, show=False)
            plt.gcf().set_size_inches(8,6); plt.gcf().set_dpi(200); plt.tight_layout()
            plt.savefig(os.path.join(outdir, "shap_waterfall.png"), bbox_inches="tight", dpi=300)
            plt.savefig(os.path.join(outdir, "shap_waterfall.pdf"), bbox_inches="tight")
            plt.close()
        except Exception as e2:
            print(f"[SHAP] waterfall fallback failed: {e2}")

    # Bundle SHAP figures into a single PDF
    try:
        bundle = os.path.join(outdir, "shap_plots.pdf")
        with PdfPages(bundle) as pdf:
            for fn in ["shap_summary.png", "shap_beeswarm.png", "shap_force.png", "shap_waterfall.png"]:
                path = os.path.join(outdir, fn)
                if os.path.exists(path):
                    img = plt.imread(path)
                    plt.figure(figsize=(8,6), dpi=200)
                    plt.imshow(img); plt.axis('off')
                    pdf.savefig(bbox_inches='tight'); plt.close()
    except Exception:
        pass
        
        
def filter_features_to_pcga(df_feat: pd.DataFrame, pcga_set: set[str]) -> pd.DataFrame:
    """Keep columns that are covariates (*_cov or *_cov_* dummies) OR whose mapped gene ∈ pcga_set.
    Uses _feature_to_gene() to infer the gene from the feature name.
    """
    cols = df_feat.columns.tolist()
    keep = []
    for c in cols:
        if c.endswith('_cov') or ('_cov_' in c):  # keep covariates
            keep.append(c)
            continue
        g = _feature_to_gene(c)
        if g in pcga_set:
            keep.append(c)
    return df_feat.reindex(columns=keep)

def collapse_gene_abnormalities(X_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse columns to gene-level abnormality: for each gene, value=any(col>0) across its feature columns.
    Keeps covariate columns (*_cov and *_cov_* dummies) unchanged.
    """
    if X_df.empty:
        print("[WARN] collapse_gene_abnormalities: Received empty DataFrame, returning as-is")
        return X_df
    cov_cols = [c for c in X_df.columns if c.endswith('_cov') or ('_cov_' in c)]
    feat_cols = [c for c in X_df.columns if c not in cov_cols]
    # Group non-covariate columns by gene symbol
    gene_to_cols: dict[str, list[str]] = {}
    for c in feat_cols:
        g = _feature_to_gene(c)
        gene_to_cols.setdefault(g, []).append(c)
    out = {}
    for g, cols in gene_to_cols.items():
        sub = X_df[cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        out[g] = (sub.values > 0).any(axis=1).astype(int)
    G = pd.DataFrame(out, index=X_df.index) if len(out)>0 else pd.DataFrame(index=X_df.index)
    # Append covariates (unaltered)
    if len(cov_cols) > 0:
        G = pd.concat([G, X_df[cov_cols]], axis=1)
    return G

def collapse_delta_by_gene(delta_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse Δ features at gene-level and direction.
    Input columns like 'TP53_mut__gain', 'TP53_del__loss' -> output 'TP53__gain', 'TP53__loss' (any over types).
    """
    if delta_df.empty:
        return delta_df
    gain_map: dict[str, list[str]] = {}
    loss_map: dict[str, list[str]] = {}
    for c in delta_df.columns:
        base = str(c)
        if '__gain' in base:
            stem = base.split('__gain')[0]
            gene = _feature_to_gene(stem)
            gain_map.setdefault(gene, []).append(c)
        elif '__loss' in base:
            stem = base.split('__loss')[0]
            gene = _feature_to_gene(stem)
            loss_map.setdefault(gene, []).append(c)
        else:
            # keep unmatched columns verbatim by treating them as loss to preserve column
            loss_map.setdefault(base, []).append(c)
    out = {}
    # Aggregate gains
    for g, cols in gain_map.items():
        sub = delta_df[cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        out[f"{g}__gain"] = (sub.values > 0).any(axis=1).astype(int)
    # Aggregate losses
    for g, cols in loss_map.items():
        sub = delta_df[cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        out[f"{g}__loss"] = (sub.values > 0).any(axis=1).astype(int)
    return pd.DataFrame(out, index=delta_df.index)

# ---------------------------
# Delta feature analysis
# ---------------------------

def build_delta_matrix(df: pd.DataFrame, target_col: str, drop_cols: List[str], seq_method_col: str = 'seq_method', pcga_set: set[str] | None = PCGA_V3_SET, force_pcga_mask: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Return (Δ-feature DF, patient-level y).
    - 入力 df から target と drop_cols を除いた列を候補とする
    - 非数値列は除外
    - 0/1 以外の数値は !=0 を 1 に二値化（例：CNVの閾値化）
    - FNA と tumor のペアで gain/loss を作る
    """
    # Keep platform column for delta masking logic even if listed in drop_cols
    _drop = [c for c in drop_cols if c in df.columns and c != seq_method_col]
    feat_df_raw = df.drop(columns=_drop + [target_col], errors="ignore").copy()

    # 数値に強制変換（非数値は NaN）
    feat_num = feat_df_raw.apply(pd.to_numeric, errors="coerce")

    # すべて NaN の列は落とす
    feat_num = feat_num.loc[:, feat_num.notna().any(axis=0)]

    # 0/1 以外の列は二値化（!=0 を 1 とする）※必要に応じてしきい値は調整
    # すでに 0/1 の列はそのまま
    def _binarize(col: pd.Series) -> pd.Series:
        vals = col.dropna().unique()
        if set(vals).issubset({0, 1}):
            return col.fillna(0).astype(int)
        # 例：-1/0/1 や 実数のCN係数 → 0/1化
        return (col.fillna(0).astype(float) != 0).astype(int)

    feat_bin = feat_num.apply(_binarize, axis=0)
    # ★ サンプル名をインデックスにする（重複しない）
    feat_bin = feat_bin.copy()
    feat_bin["sample"] = df["sample"].astype(str).values
    feat_bin = feat_bin.set_index("sample")

    df2 = df.copy()
    df2["patient_id"] = df2["sample"].apply(base_id)
    # Platform flags for PCGA_v3 per row
    if seq_method_col in df2.columns:
        df2[seq_method_col] = df2[seq_method_col].astype(str)
        df2["is_pcga_row"] = df2[seq_method_col].str.upper().eq("PCGA_V3")
    else:
        df2["is_pcga_row"] = False

    # 患者ラベルの決定：FNA行を優先し、次にその他の行から最初の有効値（数値化可能）を採用
    df2["is_fna"] = df2["sample"].str.endswith("_FNA")
    def _pick_label(g: pd.DataFrame):
        vals = pd.to_numeric(g[target_col], errors="coerce")
        # FNA優先で有効値を探す
        if g["is_fna"].any():
            v = vals[g["is_fna"]].dropna()
            if len(v) > 0:
                # Validate consistency if multiple FNA rows exist
                if len(v) > 1 and v.nunique() > 1:
                    print(f"[WARN] Patient {g['patient_id'].iloc[0]}: Multiple FNA rows with different labels {v.unique()}. Using first: {v.iloc[0]}")
                return v.iloc[0]
        v = vals.dropna()
        if len(v) > 0:
            # Log warning if using non-FNA label
            if g["is_fna"].any():
                print(f"[WARN] Patient {g['patient_id'].iloc[0]}: No valid FNA label, using non-FNA label: {v.iloc[0]}")
            return v.iloc[0]
        return np.nan
    y_pat = df2.groupby("patient_id", group_keys=False).apply(_pick_label)
    y_pat = y_pat.dropna().astype(int)

    # FNA / tumor をサンプル名で抽出し、患者IDを index に差し替える
    fna_df = df2[df2["sample"].str.endswith("_FNA")][["sample", "patient_id"]].copy()
    tum_df = df2[df2["sample"].str.endswith("_tumor")][["sample", "patient_id"]].copy()

    # 共通患者ID（かつラベルが存在する患者に限定）
    common_ids = (
        pd.Index(fna_df["patient_id"].astype(str).unique())
        .intersection(pd.Index(tum_df["patient_id"].astype(str).unique()))
        .intersection(pd.Index(y_pat.index.astype(str)))
    )
    if common_ids.empty:
        raise SystemExit("[ERROR] No paired patients with both _FNA and _tumor.")

    # 同じ特徴空間で取り出し（feat_bin は sample index）
    feat_cols = feat_bin.columns  # すべてバイナリ化された特徴列
    fna_feat = feat_bin.loc[fna_df["sample"], feat_cols].copy()
    fna_feat.index = fna_df["patient_id"].values
    tum_feat = feat_bin.loc[tum_df["sample"], feat_cols].copy()
    tum_feat.index = tum_df["patient_id"].values

    # 共通患者でそろえる（ここで index は patient_id で一意）
    fna_feat = fna_feat.reindex(index=common_ids).fillna(0).astype(int)
    tum_feat = tum_feat.reindex(index=common_ids).fillna(0).astype(int)

    # === PCGA_v3 masking ===
    #  - default: mask only pairs where either specimen is PCGA_v3
    #  - force_pcga_mask=True: mask ALL pairs (restrict analysis to PCGA genes globally)
    if pcga_set is not None and len(pcga_set) > 0:
        if force_pcga_mask:
            pcga_cases_idx = list(common_ids)
        else:
            has_pcga_case = df2.groupby("patient_id")["is_pcga_row"].any()
            pcga_cases_idx = [pid for pid in common_ids if bool(has_pcga_case.get(pid, False))]
        if len(pcga_cases_idx) > 0:
            feat_cols = fna_feat.columns.tolist()
            non_pcga_cols = [c for c in feat_cols if _feature_to_gene(c) not in pcga_set]
            if len(non_pcga_cols) > 0:
                fna_feat.loc[pcga_cases_idx, non_pcga_cols] = 0
                tum_feat.loc[pcga_cases_idx, non_pcga_cols] = 0

    # gain / loss
    gains = ((fna_feat == 0) & (tum_feat == 1)).astype(int)
    gains.columns = [f"{c}__gain" for c in gains.columns]
    losses = ((fna_feat == 1) & (tum_feat == 0)).astype(int)
    losses.columns = [f"{c}__loss" for c in losses.columns]

    delta_df = pd.concat([gains, losses], axis=1)
    y_pat = y_pat.loc[common_ids]
    return delta_df, y_pat


def fisher_scan(delta_df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    rows = []
    for col in delta_df.columns:
        x = delta_df[col].values.astype(int)
        yv = y.values.astype(int)
        a = int(((x==1) & (yv==1)).sum())
        b = int(((x==1) & (yv==0)).sum())
        c = int(((x==0) & (yv==1)).sum())
        d = int(((x==0) & (yv==0)).sum())
        OR, p = (np.nan, np.nan)
        if SCIPY_OK:
            try:
                (OR, p) = fisher_exact([[a, b], [c, d]], alternative="two-sided")
            except Exception:
                OR, p = np.nan, np.nan
        rows.append(dict(feature=col, a=a, b=b, c=c, d=d, OR=OR, p=p))
    return pd.DataFrame(rows).sort_values("p", na_position="last")


def select_covariates_l1(X_df: pd.DataFrame, cov_cols: list, y: pd.Series, C: float = 1.0, max_k: int | None = None) -> pd.DataFrame:
    """Run L1-logistic only on the provided covariate columns to select a sparse subset.
    Returns a DataFrame with columns [feature, coef] sorted by |coef| desc. If max_k is given,
    keep only the top-k by |coef| (non-zero)."""
    if len(cov_cols) == 0:
        return pd.DataFrame({"feature": [], "coef": []})
    Xc = X_df[cov_cols].copy()
    # All modeling columns are numeric; still, make sure dtype is float
    Xc = Xc.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    pipe = Pipeline([
        ("sc", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(penalty="l1", solver="saga", max_iter=5000, C=C, class_weight="balanced"))
    ])
    pipe.fit(Xc.values, y.values.astype(int))
    coefs = pipe.named_steps["clf"].coef_.ravel()
    sel = np.where(coefs != 0)[0]
    df = pd.DataFrame({
        "feature": [cov_cols[i] for i in sel] if len(sel)>0 else [],
        "coef": coefs[sel] if len(sel)>0 else []
    }).sort_values("coef", key=np.abs, ascending=False)
    if max_k is not None and len(df) > max_k:
        df = df.iloc[:max_k].copy()
    return df

def l1_multivariate(X: pd.DataFrame, y: pd.Series, C: float = 1.0) -> pd.DataFrame:
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("sc", StandardScaler(with_mean=True, with_std=True)),
        ("clf", LogisticRegression(
            penalty="l1", solver="saga", max_iter=5000, C=C, class_weight="balanced"))
    ])
    pipe.fit(X.values, y.values.astype(int))
    coefs = pipe.named_steps["clf"].coef_.ravel()
    sel = np.where(coefs != 0)[0]
    if len(sel) == 0:
        print("[WARN] l1_multivariate: No features selected with L1 regularization. Returning empty DataFrame.")
    top = pd.DataFrame({
        "feature": X.columns[sel] if len(sel)>0 else [],
        "coef": coefs[sel] if len(sel)>0 else []
    })
    if len(top) > 0:
        top = top.sort_values("coef", key=np.abs, ascending=False)
    return top

# --- FNA univariate association helper ---
def fna_univariate_tests(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Univariate association for FNA features vs binary y (CAP).
    - Binary (0/1) columns: Fisher's exact (OR, p)
    - Continuous columns: Mann-Whitney U (location shift) with p; also report medians.
    Adds BH-FDR (q) for all tests together.
    """
    rows = []
    m = X.shape[1]
    for col in X.columns:
        s = pd.to_numeric(X[col], errors='coerce')
        s0 = s[y==0].dropna(); s1 = s[y==1].dropna()
        # decide binary
        vals = pd.unique(s.dropna())
        is_bin = set(pd.Series(vals).dropna().astype(float).unique()).issubset({0.0,1.0})
        if is_bin:
            a = int(((s==1) & (y==1)).sum())
            b = int(((s==1) & (y==0)).sum())
            c = int(((s==0) & (y==1)).sum())
            d = int(((s==0) & (y==0)).sum())
            OR, p = (np.nan, np.nan)
            if SCIPY_OK:
                try:
                    OR, p = fisher_exact([[a,b],[c,d]], alternative="two-sided")
                except Exception:
                    OR, p = (np.nan, np.nan)
            rows.append(dict(feature=col, test="fisher", OR=OR, a=a, b=b, c=c, d=d,
                              median0=np.nan, median1=np.nan, diff_median=np.nan, p=p))
        else:
            p_mw = np.nan
            med0 = np.nanmedian(s0) if s0.size else np.nan
            med1 = np.nanmedian(s1) if s1.size else np.nan
            if SCIPY_OK:
                try:
                    p_mw = mannwhitneyu(s1, s0, alternative="two-sided").pvalue
                except Exception:
                    pass
            rows.append(dict(feature=col, test="mwu", OR=np.nan, a=np.nan, b=np.nan, c=np.nan, d=np.nan,
                              median0=med0, median1=med1, diff_median=(med1-med0), p=p_mw))
    dfu = pd.DataFrame(rows)
    # Benjamini-Hochberg FDR
    if len(dfu)>0 and dfu['p'].notna().any():
        pvals = dfu['p'].values.astype(float)
        m = np.isfinite(pvals).sum()
        order = np.argsort(np.where(np.isfinite(pvals), pvals, np.inf))
        rank = np.empty_like(order); rank[order] = np.arange(1, len(order)+1)
        q = np.full_like(pvals, fill_value=np.nan, dtype=float)
        # compute BH only for finite p
        finite = np.isfinite(pvals)
        if finite.any():
            ps = pvals[finite]
            ord_idx = np.argsort(ps)
            ps_sorted = ps[ord_idx]
            q_sorted = ps_sorted * m / (np.arange(1, ps_sorted.size+1))
            # monotone decreasing adjustment
            for i in range(q_sorted.size-2, -1, -1):
                q_sorted[i] = min(q_sorted[i], q_sorted[i+1])
            q_vals = np.full(ps.shape, np.nan)
            q_vals[ord_idx] = q_sorted
            q[finite] = q_vals
        dfu['q_bh'] = q
    return dfu.sort_values('p', na_position='last')


# ---------------------------
# Tumor GSVA analysis (mechanism only)
# ---------------------------

def load_gsva_tumor(gsva_path: str) -> pd.DataFrame:
    # Accept TSV with first column as sample (or index)
    df = pd.read_csv(gsva_path, sep="\t")
    if "sample" not in df.columns:
        # assume first column is sample id
        df = df.rename(columns={df.columns[0]: "sample"})
    return df

def tumor_gsva_mechanism(gsva_df: pd.DataFrame, main_df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Join tumor-only rows by sample
    tum = main_df[main_df["sample"].str.endswith("_tumor")][["sample", target_col]].copy()
    merge = tum.merge(gsva_df, on="sample", how="inner")
    if merge.empty:
        raise SystemExit("[WARN] No overlap between tumor samples and GSVA file.")

    y = merge[target_col].astype(int).values
    feat_cols = [c for c in merge.columns if c not in ("sample", target_col)]
    X = merge[feat_cols].astype(float)

    # Univariate tests (Mann-Whitney; fallback to t-test)
    rows = []
    for col in feat_cols:
        x1 = X[col][merge[target_col]==1].values
        x0 = X[col][merge[target_col]==0].values
        p_mw = np.nan; p_tt = np.nan
        if SCIPY_OK:
            try:
                p_mw = mannwhitneyu(x1, x0, alternative="two-sided").pvalue
                p_tt = ttest_ind(x1, x0, equal_var=False).pvalue
            except Exception:
                pass
        rows.append(dict(pathway=col, p_mannwhitney=p_mw, p_ttest=p_tt,
                         mean_pos=np.nanmean(x1) if x1.size else np.nan,
                         mean_neg=np.nanmean(x0) if x0.size else np.nan))
    uni = pd.DataFrame(rows).sort_values("p_mannwhitney", na_position="last")

    # Multivariate sparse logistic
    top = l1_multivariate(X, merge[target_col])
    return uni, top
    
def _compute_auc_brier(y_true: np.ndarray, p_pred: np.ndarray) -> tuple[float, float]:
    try:
        auc = roc_auc_score(y_true, p_pred)
    except Exception:
        auc = np.nan
    try:
        brier = brier_score_loss(y_true, p_pred)
    except Exception:
        brier = np.nan
    return float(auc), float(brier)

def apparent_performance_dietnet(X: np.ndarray, y: np.ndarray, hidden: List[int], epochs: int, lr: float, w_l1: float, seed: int, device=None) -> tuple[float, float]:
    """Fit DietNet on ALL data and return in-sample (apparent) AUC and Brier."""
    pred_fn, _U = dietnet_fit_full(X, y, hidden=hidden, epochs=epochs, lr=lr, w_l1=w_l1, seed=seed, device=device)
    p = pred_fn(X.astype("float32", copy=False))
    auc, brier = _compute_auc_brier(y.astype(int), p)
    return auc, brier

def bootstrap_optimism_dietnet(X: np.ndarray, y: np.ndarray, B: int, hidden: List[int], epochs: int, lr: float, w_l1: float, seed: int, device=None) -> tuple[pd.DataFrame, dict]:
    """Harrell's bootstrap optimism correction.
    For b=1..B: sample n with replacement, fit model on bootstrap sample, compute:
      - apparent_b: performance on bootstrap sample
      - test_b: performance when applying that model to the ORIGINAL dataset
    optimism_b = apparent_b - test_b.
    """
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    rows = []
    for b in range(1, B+1):
        idx = rng.choice(n, size=n, replace=True)
        Xb = X[idx]; yb = y[idx]
        # fit on bootstrap
        pred_fn, _U = dietnet_fit_full(Xb, yb, hidden=hidden, epochs=epochs, lr=lr, w_l1=w_l1, seed=seed+b, device=device)
        # apparent on bootstrap
        pb_boot = pred_fn(Xb.astype("float32", copy=False))
        auc_app, brier_app = _compute_auc_brier(yb.astype(int), pb_boot)
        # test on original
        pb_org = pred_fn(X.astype("float32", copy=False))
        auc_tst, brier_tst = _compute_auc_brier(y.astype(int), pb_org)
        rows.append(dict(b=b, auc_app=auc_app, auc_test_on_orig=auc_tst,
                         auc_optimism=(auc_app - auc_tst),
                         brier_app=brier_app, brier_test_on_orig=brier_tst,
                         brier_optimism=(brier_app - brier_tst)))
    df = pd.DataFrame(rows)
    # summary（平均の楽観バイアス）
    summary = dict(
        auc_apparent_full=np.nan,   # 呼び出し側で埋める
        brier_apparent_full=np.nan,
        auc_optimism_mean=float(np.nanmean(df['auc_optimism'])),
        brier_optimism_mean=float(np.nanmean(df['brier_optimism'])),
        auc_corrected=np.nan,       # 呼び出し側で埋める
        brier_corrected=np.nan
    )
    return df, summary


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="wide CSV with columns: sample, CAP_binary, genomic features...")
    ap.add_argument("--target", default="CAP_binary", help="binary target column name (0/1)")
    ap.add_argument("--drop_cols", default="sample,seq_method,specimen_type", help="meta columns not used as genomic features")
    ap.add_argument("--covars_csv", default=None, help="optional CSV with clinical covariates (must contain 'sample')")
    ap.add_argument("--covar_cols", default=None, help="comma-separated column names in covars_csv to append as features for FNA model")
    ap.add_argument("--gsva_tumor_tsv", default=None, help="optional tumor GSVA TSV (sample + pathways) for mechanism analysis")
    ap.add_argument("--covar_select_l1", action="store_true", help="Use L1-logistic to select a sparse subset of covariate columns before modeling")
    ap.add_argument("--covar_select_C", type=float, default=1.0, help="Inverse regularization strength C for covariate L1 selection (larger = less sparse)")
    ap.add_argument("--covar_select_maxk", type=int, default=None, help="Optional cap on the number of covariates to retain after L1 selection")
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--w_l1", type=float, default=1e-4)
    ap.add_argument("--hidden", default="64,32", help="DietNet feature-MLP hidden sizes, e.g., '64,32' or '32'")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default=".", help="output directory")
    ap.add_argument("--external_csv", default=None, help="optional external cohort CSV to apply trained FNA model")
    ap.add_argument("--repeats", type=int, default=1, help="number of repeated CV runs with different seeds")
    ap.add_argument("--opt_threshold", action="store_true", help="optimize decision threshold per fold by Youden index")
    ap.add_argument("--min_feat_prevalence", type=float, default=0.0, help="drop binary features with prevalence below this (and above 1-p)")
    ap.add_argument("--max_corr", type=float, default=1.01, help="drop one of each highly correlated feature pairs (abs corr > this)")
    ap.add_argument("--topk_stability", type=int, default=100, help="top-K features per run used to compute selection frequency")
    ap.add_argument("--save_fold_metrics", action="store_true", help="save per-repetition aggregate metrics to CSV")
    ap.add_argument("--perm_runs", type=int, default=0, help="number of permutation runs for null AUC distribution (0=skip)")
    ap.add_argument("--bootstrap_optimism", type=int, default=0,
                help="Number of bootstrap resamples for optimism-corrected AUC/Brier (0=skip)")
    ap.add_argument("--calibration", action="store_true", help="compute CV calibration (Brier + calibration curve)")
    ap.add_argument("--grid_hidden", default=None, help="semicolon-separated hidden configs; e.g. '32;64;64,32'")
    ap.add_argument("--grid_w_l1", default=None, help="comma-separated list for L1 weights, e.g. '1e-5,1e-4,1e-3'")
    ap.add_argument("--grid_epochs", default=None, help="comma-separated list for epochs, e.g. '150,200,300'")
    ap.add_argument("--stability_B", type=int, default=0, help="stability selection repetitions with L1-logit (0=skip)")
    ap.add_argument("--stability_subsample", type=float, default=0.7, help="subsample fraction for stability selection")
    ap.add_argument("--stability_Cs", default="1.0", help="comma-separated C values for L1-logit in stability selection")
    ap.add_argument("--platform_col", default=None, help="column name indicating platform (e.g., seq_method)")
    ap.add_argument("--platform_adjust", choices=["none","indicator","stratify"], default="none",
                    help="platform handling: add indicator to features, or stratify CV by (y, platform)")
    ap.add_argument("--device", choices=["auto","cpu","cuda","mps"], default="auto",
                    help="compute device to use; default auto (mps>cuda>cpu)")
    ap.add_argument("--time_profile", action="store_true",
                    help="print simple timing per fold and per epoch")
    ap.add_argument("--use_batchnorm", action="store_true",
                    help="Enable batch normalization in DietNet auxiliary network (recommended for large datasets with n_train > 1000)")
    ap.add_argument("--compare_covars", action="store_true", help="Run a paired comparison: genetics-only vs genetics+covariates, output AUC/Brier/NetBenefit deltas")
    ap.add_argument("--pathway_csv", default=None, help="CSV mapping of feature->pathway for pathway-level aggregation (columns: feature,pathway)")
    ap.add_argument("--pathway_mode", choices=["any","count","mean"], default="any", help="How to aggregate features within a pathway: any(OR), count, or mean")
    ap.add_argument("--volcano_top_labels", type=int, default=15, help="Number of points to label in volcano plot")
    ap.add_argument("--restrict_to_pcga", action="store_true", help="If set, restrict genomic features and Δ analysis to PCGA_v3 genes")
    ap.add_argument("--collapse_gene_abn", action="store_true",
                help="Collapse alteration-type features into gene-level abnormality (e.g., TP53_mut/amp/del/loh -> TP53)")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    if TORCH_OK:
        dev = pick_device(args.device)
        print(f"[Torch] Using device: {dev}")

    # ---- Load main
    df = pd.read_csv(args.csv)
    if args.target not in df.columns:
        sys.exit(f"[ERROR] target column '{args.target}' not found.")
    if "sample" not in df.columns:
        sys.exit("[ERROR] input CSV must contain a 'sample' column.")
    df["sample"] = df["sample"].astype(str)
    # ---- Platform handling (optional)
    platform_series = None
    if args.platform_col and args.platform_col in df.columns:
        platform_series = df[args.platform_col].astype(str)
        # 'indicator' の場合は one-hot 化させたいので、あとで drop_cols から除外する
    else:
        args.platform_adjust = "none"

    # ---- Merge clinical covariates (optional)
    covar_basenames = []  # keep original covariate base names for compare_covars
    if args.covars_csv is not None:
        cov = pd.read_csv(args.covars_csv)
        if "sample" not in cov.columns:
            sys.exit("[ERROR] covars_csv must contain a 'sample' column.")

        # 使う列だけに絞って _cov サフィックスを付与（重複回避）
        if args.covar_cols:
            covar_list = [c.strip() for c in args.covar_cols.split(",") if c.strip()]
            covar_basenames = covar_list[:]
            keep = ["sample"] + [c for c in covar_list if c in cov.columns]
            cov = cov[keep].rename(columns={c: f"{c}_cov" for c in cov.columns if c != "sample"})
        else:
            # covar_cols 未指定なら右側の列は全部 _cov を付ける
            cov = cov.rename(columns={c: f"{c}_cov" for c in cov.columns if c != "sample"})

        df = df.merge(cov, on="sample", how="left")

    # ---- FNA-only prediction features
    is_fna = df["sample"].str.endswith("_FNA")
    df_fna = df[is_fna].copy()
    df_fna["patient_id"] = df_fna["sample"].apply(base_id)

    drop_cols = [c.strip() for c in args.drop_cols.split(",") if c.strip()]

    # ▼ サフィックス付きも含めてドロップ対象を構築（ただし *_cov 系は落とさない）
    to_drop = set()
    for c in drop_cols + [args.target]:
        if args.platform_adjust == "indicator" and args.platform_col and c == args.platform_col:
            continue  # プラットフォームは残す（one-hot化する）
        for col in df_fna.columns:
            if col == c or col.startswith(c + "_"):
                # 共変量は温存（後段の compare_covars 用）
                if col.endswith("_cov") or ("_cov_" in col):
                    continue
                to_drop.add(col)

    # ▼ 候補特徴行列を作成（FNAのみ; target/メタ列を除去）
    genomic_feat = df_fna.drop(columns=list(to_drop), errors="ignore").copy()
    # 念のため target が残っていれば除去
    genomic_feat = genomic_feat.drop(columns=[args.target], errors="ignore")
    # サンプル/患者IDは特徴に含めない
    for _mc in ["sample", "patient_id"]:
        if _mc in genomic_feat.columns:
            genomic_feat = genomic_feat.drop(columns=[_mc])
    # --- Ensure platform column is NOT used as a feature unless explicitly requested as 'indicator'
    if args.platform_col and args.platform_adjust != "indicator":
        plat = args.platform_col
        # drop raw platform column if it slipped through
        if plat in genomic_feat.columns:
            genomic_feat = genomic_feat.drop(columns=[plat])
        # also drop any pre-existing one-hot columns with the same prefix (defensive)
        plat_pref = f"{plat}_"
        drop_like = [c for c in genomic_feat.columns if c.startswith(plat_pref)]
        if drop_like:
            genomic_feat = genomic_feat.drop(columns=drop_like, errors="ignore")

    # --- Remove duplicate (non-_cov) versions of declared covariates to avoid double representation
    if len(covar_basenames) > 0:
        base_set = set([b.strip() for b in covar_basenames if b.strip()])
        base_dups = []
        for b in base_set:
            # drop base and its one-hot dummies if they exist without the _cov tag
            base_dups.extend([c for c in genomic_feat.columns
                              if (c == b or c.startswith(b + '_')) and ('_cov' not in c)])
        if base_dups:
            genomic_feat = genomic_feat.drop(columns=list(dict.fromkeys(base_dups)), errors='ignore')
            print(f"[Covar] Dropped non-_cov duplicates of declared covariates: {len(base_dups)} columns")

    # --- Try to coerce covariate columns to numeric (e.g., Age_cov) before get_dummies
    # This prevents unintended one-hot for numeric covariates read as object due to missing values
    for c in list(genomic_feat.columns):
        if c.endswith("_cov"):
            col_num = pd.to_numeric(genomic_feat[c], errors="coerce")
            # Accept coercion if at least half the rows are numeric
            if np.isfinite(col_num).sum() >= max(5, int(0.5 * len(col_num))):
                genomic_feat[c] = col_num.astype(float)

    # ▼ 残った object 列は一括で one-hot エンコード（臨床カテゴリを活かす）
    obj_cols = genomic_feat.select_dtypes(include=["object"]).columns.tolist()
    if obj_cols:
        genomic_feat = pd.get_dummies(genomic_feat, columns=obj_cols, drop_first=True)
    # --- Defensive cleanup: remove any one-hot platform dummies unless 'indicator' mode
    if args.platform_col and args.platform_adjust != "indicator":
        plat = args.platform_col
        plat_pref = f"{plat}_"
        drop_like = [c for c in genomic_feat.columns if (c == plat or c.startswith(plat_pref))]
        if drop_like:
            genomic_feat = genomic_feat.drop(columns=drop_like, errors="ignore")
    # --- Optional: restrict FNA modeling features to PCGA genes
    if args.restrict_to_pcga:
        before = genomic_feat.shape[1]
        genomic_feat = filter_features_to_pcga(genomic_feat, PCGA_V3_SET)
        after = genomic_feat.shape[1]
        try:
            with open(os.path.join(args.outdir, 'restrict_pcga_fna_kept_features.txt'), 'w') as f:
                for c in genomic_feat.columns:
                    f.write(c + '\n')
            print(f"[PCGA] FNA features restricted to PCGA genes: {before} -> {after}")
        except Exception:
            pass
    # --- Optional: collapse FNA features to gene-level abnormality
    if args.collapse_gene_abn:
        before = genomic_feat.shape[1]
        genomic_feat = collapse_gene_abnormalities(genomic_feat)
        after = genomic_feat.shape[1]
        try:
            with open(os.path.join(args.outdir, 'collapse_gene_fna_features.txt'), 'w') as f:
                for c in genomic_feat.columns:
                    f.write(c + '\n')
            print(f"[Collapse] FNA features collapsed to gene-level: {before} -> {after}")
        except Exception:
            pass
    # --- Optional: L1-based automatic covariate selection
    if args.covar_select_l1:
        # Detect covariate columns present in the current modeling matrix
        cov_cols = []
        cols = list(genomic_feat.columns)
        cov_cols.extend([c for c in cols if c.endswith('_cov') or ('_cov_' in c)])
        # also include basenames from --covar_cols in case some survived without suffix (defensive)
        if len(covar_basenames) > 0:
            base_set = set([b.strip() for b in covar_basenames if b.strip()])
            for c in cols:
                for b in base_set:
                    if c == b or c.startswith(b + '_'):
                        cov_cols.append(c)
        # Deduplicate while preserving order
        seen = set(); cov_cols = [c for c in cov_cols if not (c in seen or seen.add(c))]
        if len(cov_cols) > 0:
            X_df_tmp = genomic_feat.copy()
            y_sr_tmp = pd.Series(df_fna[args.target].astype(int).values, name='y')
            sel_df = select_covariates_l1(X_df_tmp, cov_cols, y_sr_tmp, C=args.covar_select_C, max_k=args.covar_select_maxk)
            # Persist selection summary
            try:
                sel_df.to_csv(os.path.join(args.outdir, 'covariate_selection_l1.csv'), index=False)
            except Exception:
                pass
            keep_set = set(sel_df['feature'].tolist())
            drop_list = [c for c in cov_cols if c not in keep_set]
            if len(drop_list) > 0:
                genomic_feat = genomic_feat.drop(columns=drop_list, errors='ignore')
                print(f"[Covar-Select] Dropped {len(drop_list)} covariate columns; kept {len(keep_set)}")
        else:
            print('[Covar-Select] No covariate columns detected for selection')

    # Optional pruning for robustness
    genomic_feat = prune_features(
        genomic_feat.fillna(0).astype(float),
        min_prevalence=args.min_feat_prevalence,
        max_corr=args.max_corr
    )
    feat_names = genomic_feat.columns.tolist()
    X_fna = genomic_feat.fillna(0).astype(float).values
    y_fna = df_fna[args.target].astype(int).values
    groups_fna = df_fna["patient_id"].values
    feat_names = genomic_feat.columns.tolist()

    # ---- Part 1: FNA-only DietNet CV（患者IDで層化分割）
    if not TORCH_OK:
        print("[WARN] PyTorch not available; Part 1 (DietNet) will be skipped.")
    else:
        hidden = parse_hidden(args.hidden)

        # Repeated CV
        all_auc_m, all_acc_m, all_sen_m, all_spe_m = [], [], [], []
        all_thr = []
        feat_import_runs = []
        rep_rows = []
        for rep in range(args.repeats):
            seed_rep = args.seed + rep
            # If stratify by (y, platform) is requested, build a combined label
            y_for_strat = y_fna
            groups_for_cv = groups_fna
            if args.platform_adjust == "stratify" and args.platform_col:
                # FNA行におけるプラットフォーム値（df_fnaから取得）
                plat_col_name = args.platform_col if args.platform_col in df_fna.columns else (args.platform_col + "_cov" if (args.platform_col + "_cov") in df_fna.columns else None)
                if plat_col_name is not None:
                    plat = df_fna[plat_col_name].astype(str).values
                    levels = {v:i for i,v in enumerate(pd.unique(plat))}
                    plat_code = np.array([levels[v] for v in plat], dtype=int)
                    y_for_strat = y_fna * 10 + plat_code  # (y, platform) を同時に層化
            (auc_m, auc_s, acc_m, acc_s, sen_m, sen_s, spe_m, spe_s, feat_score, thr_arr) = dietnet_cv(
                X=X_fna, y=y_fna,
                kfold=args.kfold, hidden=hidden,
                epochs=args.epochs, lr=args.lr, w_l1=args.w_l1, seed=seed_rep,
                groups=groups_for_cv, opt_threshold=args.opt_threshold,
                y_strat=(y_for_strat if args.platform_adjust == "stratify" else None),
                device=pick_device(args.device), time_profile=args.time_profile,
                use_batchnorm=args.use_batchnorm
            )
            all_auc_m.append(auc_m); all_acc_m.append(acc_m); all_sen_m.append(sen_m); all_spe_m.append(spe_m)
            if len(thr_arr) > 0:
                all_thr.extend(thr_arr.tolist())
            feat_import_runs.append(feat_score)
            if args.save_fold_metrics:
                rep_rows.append(dict(rep=rep, auc_mean=auc_m, acc_mean=acc_m, sens_mean=sen_m, spec_mean=spe_m))

        def _msd(x):
            return (np.nanmean(x), np.nanstd(x))
        aucM, aucS = _msd(all_auc_m); accM, accS = _msd(all_acc_m); senM, senS = _msd(all_sen_m); speM, speS = _msd(all_spe_m)
        with open(os.path.join(args.outdir, "fna_cv_summary.txt"), "w") as f:
            f.write(
                f"AUC: {aucM:.3f} ± {aucS:.3f}\n"
                f"ACC: {accM:.3f} ± {accS:.3f}\n"
                f"Sens: {senM:.3f} ± {senS:.3f}\n"
                f"Spec: {speM:.3f} ± {speS:.3f}\n"
                + (f"Threshold (Youden) median: {np.nanmedian(all_thr):.3f}\n" if len(all_thr)>0 else "")
            )

        feat_import_runs = np.vstack(feat_import_runs) if len(feat_import_runs)>0 else np.zeros((1, X_fna.shape[1]))
        mean_import = feat_import_runs.mean(axis=0)
        topk = max(1, int(args.topk_stability))
        sel_freq = np.zeros_like(mean_import)
        for r in range(feat_import_runs.shape[0]):
            idx_top = np.argsort(-feat_import_runs[r])[:topk]
            sel_freq[idx_top] += 1
        sel_freq = sel_freq / max(1, feat_import_runs.shape[0])

        df_rank = pd.DataFrame({
            "feature": feat_names,
            "mean_importance": mean_import,
            "selection_frequency": sel_freq
        }).sort_values(["selection_frequency", "mean_importance"], ascending=[False, False])
        df_rank.to_csv(os.path.join(args.outdir, "fna_feature_stability.csv"), index=False)

        idx = np.argsort(-mean_import)[:args.topk_stability]
        pd.DataFrame({"feature": [feat_names[i] for i in idx], "importance": mean_import[idx]}).to_csv(
            os.path.join(args.outdir, "fna_top_features.csv"), index=False
        )

        if args.save_fold_metrics and len(rep_rows)>0:
            pd.DataFrame(rep_rows).to_csv(os.path.join(args.outdir, "fna_cv_repeats_summary.csv"), index=False)

        print("[Part1] Saved fna_cv_summary.txt, fna_top_features.csv and fna_feature_stability.csv")
        
        # === (Optional) Bootstrap optimism correction for AUC/Brier ===
        if args.bootstrap_optimism > 0:
            try:
                # apparent on full data
                auc_app_full, brier_app_full = apparent_performance_dietnet(
                    X_fna.astype(np.float32), y_fna.astype(int),
                    hidden=parse_hidden(args.hidden), epochs=args.epochs,
                    lr=args.lr, w_l1=args.w_l1, seed=args.seed, device=pick_device(args.device)
                )
                # bootstrap optimism
                df_boot, summ = bootstrap_optimism_dietnet(
                    X_fna.astype(np.float32), y_fna.astype(int), B=int(args.bootstrap_optimism),
                    hidden=parse_hidden(args.hidden), epochs=args.epochs,
                    lr=args.lr, w_l1=args.w_l1, seed=args.seed, device=pick_device(args.device)
                )
                summ['auc_apparent_full'] = float(auc_app_full)
                summ['brier_apparent_full'] = float(brier_app_full)
                summ['auc_corrected'] = float(auc_app_full - summ['auc_optimism_mean'])
                summ['brier_corrected'] = float(brier_app_full - summ['brier_optimism_mean'])
                # save
                df_boot.to_csv(os.path.join(args.outdir, 'fna_bootstrap_optimism_runs.csv'), index=False)
                with open(os.path.join(args.outdir, 'fna_bootstrap_optimism_summary.txt'), 'w') as f:
                    f.write(f"AUC apparent (full): {summ['auc_apparent_full']:.4f}\n")
                    f.write(f"AUC optimism (mean): {summ['auc_optimism_mean']:.4f}\n")
                    f.write(f"AUC optimism-corrected: {summ['auc_corrected']:.4f}\n")
                    f.write(f"Brier apparent (full): {summ['brier_apparent_full']:.4f}\n")
                    f.write(f"Brier optimism (mean): {summ['brier_optimism_mean']:.4f}\n")
                    f.write(f"Brier optimism-corrected: {summ['brier_corrected']:.4f}\n")
                print('[Bootstrap] Saved fna_bootstrap_optimism_runs.csv and fna_bootstrap_optimism_summary.txt')
            except Exception as e:
                print(f'[Bootstrap] Optimism correction failed: {e}')

        # === FNA genetic features: univariate & sparse multivariate associations (for CAP) ===
        try:
            X_df = pd.DataFrame(X_fna, columns=feat_names)
            y_sr = pd.Series(y_fna, name='y')
            uni_fna = fna_univariate_tests(X_df, y_sr)
            uni_fna.to_csv(os.path.join(args.outdir, 'fna_univariate.csv'), index=False)
            top_fna = l1_multivariate(X_df, y_sr, C=1.0)
            top_fna.to_csv(os.path.join(args.outdir, 'fna_multivariate_top.csv'), index=False)
            print('[Assoc] Saved fna_univariate.csv and fna_multivariate_top.csv')
            try:
                make_volcano_from_univariate(os.path.join(args.outdir,'fna_univariate.csv'), args.outdir, top_labels=args.volcano_top_labels)
                if not os.path.exists(os.path.join(args.outdir, 'fna_volcano.png')):
                    print('[Plot] Volcano did not produce a file (e.g., all q=1 or no usable x).')
            except Exception as e:
                print(f'[Plot] Volcano skipped: {e}')
        except Exception as e:
            print(f'[Assoc] Skipped FNA univariate/multivariate due to error: {e}')

        # === CV ROC (thin per fold + mean) and 95% CI ===
        fold_preds, fold_aucs = collect_cv_predictions(
            X_fna, y_fna, args.kfold, hidden=parse_hidden(args.hidden), epochs=args.epochs,
            lr=args.lr, w_l1=args.w_l1, seed=args.seed, groups=groups_fna,
            y_strat=(y_for_strat if args.platform_adjust == "stratify" else None),
            device=pick_device(args.device), time_profile=args.time_profile
        )
        roc_png = os.path.join(args.outdir, "fna_cv_roc.png")
        auc_m_plot, ci_l, ci_h = plot_cv_roc_with_ci(fold_preds, out_png=roc_png, title_prefix="ROC (CV)")

        # Aggregate predictions across folds for DCA & calibration-esque summaries
        if len(fold_preds) > 0:
            y_all = np.concatenate([it[0] for it in fold_preds])
            p_all = np.concatenate([it[1] for it in fold_preds])
            thr, nb, nb_all = decision_curve(y_all, p_all, thresholds=np.linspace(0.1,0.5,41))
            dca_png = os.path.join(args.outdir, "fna_dca.png")
            plot_decision_curve(thr, nb, nb_all, out_png=dca_png)

        # === Optional: compare genetics-only vs genetics+covariates ===
        if args.compare_covars:
            try:
                # Robust detection of covariate columns present in the modeling matrix
                cov_cols = []
                cols = list(genomic_feat.columns)
                # (a) suffix-based (_cov and _cov_ one-hot)
                cov_cols.extend([c for c in cols if c.endswith('_cov') or ('_cov_' in c)])
                # (b) basename-based (unsuffixed columns & their dummies)
                if len(covar_basenames) > 0:
                    base_set = set([b.strip() for b in covar_basenames if b.strip()])
                    for c in cols:
                        for b in base_set:
                            if c == b or c.startswith(b + '_'):
                                cov_cols.append(c)
                # deduplicate while preserving order
                seen = set(); cov_cols = [c for c in cov_cols if not (c in seen or seen.add(c))]
                try:
                    dbg_path = os.path.join(args.outdir, 'covariate_columns_detected_for_compare.txt')
                    with open(dbg_path, 'w') as fdbg:
                        for c in cov_cols:
                            fdbg.write(c + '\n')
                    print(f"[Compare] Detected {len(cov_cols)} covariate columns (saved to {dbg_path})")
                except Exception:
                    pass
                if len(cov_cols)==0:
                    print('[Compare] No covariate columns found; skip compare_covars')
                else:
                    # genetics+covariates (current X_fna)
                    y_all_cov, p_all_cov = oof_from_cv(X_fna.astype(float), y_fna.astype(int), args.kfold, parse_hidden(args.hidden), args.epochs, args.lr, args.w_l1, args.seed, groups_fna,
                                                       (y_for_strat if args.platform_adjust == 'stratify' else None), pick_device(args.device))
                    met_cov = summarize_oof(y_all_cov, p_all_cov)
                    # genetics-only: drop covariates columns
                    X_gen = pd.DataFrame(X_fna, columns=feat_names).drop(columns=cov_cols, errors='ignore').values
                    y_all_gen, p_all_gen = oof_from_cv(X_gen.astype(float), y_fna.astype(int), args.kfold, parse_hidden(args.hidden), args.epochs, args.lr, args.w_l1, args.seed, groups_fna,
                                                       y_strat=(y_fna if args.platform_adjust!='stratify' else y_for_strat), device=pick_device(args.device))
                    met_gen = summarize_oof(y_all_gen, p_all_gen)
                    comp = pd.DataFrame([
                        dict(model='genetics_only', **met_gen),
                        dict(model='genetics_plus_covars', **met_cov)
                    ])
                    # Safe indexing with validation
                    cov_row = comp.loc[comp['model']=='genetics_plus_covars']
                    gen_row = comp.loc[comp['model']=='genetics_only']
                    if len(cov_row) > 0 and len(gen_row) > 0:
                        comp['delta_auc'] = cov_row['auc'].values[0] - gen_row['auc'].values[0]
                        comp['delta_brier'] = cov_row['brier'].values[0] - gen_row['brier'].values[0]
                        comp['delta_nb_gain'] = cov_row['nb_gain_avg'].values[0] - gen_row['nb_gain_avg'].values[0]
                    else:
                        comp['delta_auc'] = np.nan
                        comp['delta_brier'] = np.nan
                        comp['delta_nb_gain'] = np.nan
                    comp.to_csv(os.path.join(args.outdir,'model_compare_summary.csv'), index=False)
                    print('[Compare] Saved model_compare_summary.csv')
            except Exception as e:
                print(f'[Compare] Skipped due to error: {e}')

        # === Optional: pathway-level aggregation ===
        if args.pathway_csv is not None:
            try:
                X_df_full = pd.DataFrame(X_fna, columns=feat_names)
                P = build_pathway_composites(X_df_full, args.pathway_csv, mode=args.pathway_mode)
                # Univariate association on pathways vs CAP
                y_sr = pd.Series(y_fna, name='y')
                uni_pw = fna_univariate_tests(P, y_sr)
                uni_pw.rename(columns={'feature':'pathway'}, inplace=True)
                uni_pw.to_csv(os.path.join(args.outdir,'pathway_univariate.csv'), index=False)
                # Simple bar of top pathways by -log10(q)
                try:
                    top = uni_pw.copy(); top['minuslog10q'] = -np.log10(pd.to_numeric(top.get('q_bh', np.nan), errors='coerce').clip(lower=1e-300))
                    top = top.sort_values('minuslog10q', ascending=False).head(20)
                    plt.figure(figsize=(7,6), dpi=200)
                    plt.barh(top['pathway'], top['minuslog10q'])
                    plt.gca().invert_yaxis()
                    plt.xlabel('-log10(FDR q)')
                    plt.title('Top pathways associated with CAP (univariate)')
                    plt.tight_layout()
                    plt.savefig(os.path.join(args.outdir,'pathway_top_bar.png'), bbox_inches='tight', dpi=300)
                    plt.savefig(os.path.join(args.outdir,'pathway_top_bar.pdf'), bbox_inches='tight')
                    plt.close()
                except Exception:
                    pass
                print('[Pathway] Saved pathway_univariate.csv and figures')
            except SystemExit as e:
                print(str(e))
            except Exception as e:
                print(f'[Pathway] Skipped due to error: {e}')

        # === Waterfall plot for RECIST (uses OOF predictions) ===
        try:
            if 'RECIST' in df_fna.columns and len(fold_preds) > 0:
                # Build OOF table with sample names
                sample_names = df_fna['sample'].astype(str).values
                rows = []
                for it in fold_preds:
                    y_te, p_te, te_idx = it[0], it[1], it[2]
                    for j, idx in enumerate(te_idx):
                        rows.append({
                            'sample': sample_names[idx],
                            'y_true': int(y_te[j]),
                            'prob_poor': float(p_te[j])
                        })
                oof_df = pd.DataFrame(rows)
                # Merge RECIST (assumed to be % change; negative = shrinkage)
                recist_df = df_fna[['sample', 'RECIST']].copy()
                out_wf = oof_df.merge(recist_df, on='sample', how='left')
                # Decide threshold to color bars: use median Youden across folds if available
                thr_use = np.nan
                try:
                    with open(os.path.join(args.outdir, 'fna_cv_summary.txt')) as f:
                        lines = f.readlines()
                    thr_lines = [l for l in lines if l.startswith('Threshold (Youden) median:')]
                    if len(thr_lines) > 0:
                        thr_use = float(thr_lines[0].split(':')[-1].strip())
                except Exception:
                    thr_use = np.nan
                if not np.isfinite(thr_use):
                    thr_use = 0.5
                out_wf['model_pred'] = (out_wf['prob_poor'] >= thr_use).astype(int)
                # Save table
                out_wf.to_csv(os.path.join(args.outdir, 'recist_waterfall_oof.csv'), index=False)
                # Plot waterfall
                df_plot = out_wf.dropna(subset=['RECIST']).sort_values('RECIST')
                import matplotlib.pyplot as plt
                fig = plt.figure(figsize=(10, 5), dpi=200)
                ax = plt.gca()
                colors = df_plot['model_pred'].map({0: 'tab:blue', 1: 'tab:red'}).values
                ax.bar(range(len(df_plot)), df_plot['RECIST'].values, color=colors)
                ax.axhline(0, color='black', linestyle='--', linewidth=1)
                ax.axhline(-30, color='gray', linestyle=':', linewidth=1)
                ax.axhline(20, color='gray', linestyle=':', linewidth=1)
                ax.set_ylabel('Best change from baseline (%)')
                ax.set_xlabel('Patients (sorted by RECIST)')
                ax.set_title(f'Waterfall plot (RECIST) with model predictions\nthreshold={thr_use:.2f} (red=pred poor, blue=pred sensitive)')
                plt.tight_layout()
                plt.savefig(os.path.join(args.outdir, 'recist_waterfall.png'), bbox_inches='tight', dpi=300)
                plt.savefig(os.path.join(args.outdir, 'recist_waterfall.pdf'), bbox_inches='tight')
                plt.close(fig)
                print('[RECIST] Saved recist_waterfall.png and recist_waterfall_oof.csv')
        except Exception as e:
            print(f'[RECIST] Waterfall generation skipped due to error: {e}')

        # Optional small PDF bundle
        try:
            pdf_path = os.path.join(args.outdir, "fna_cv_plots.pdf")
            with PdfPages(pdf_path) as pdf:
                if os.path.exists(roc_png):
                    img = plt.imread(roc_png)
                    plt.figure(figsize=(6,6), dpi=200); plt.imshow(img); plt.axis('off'); pdf.savefig(bbox_inches='tight'); plt.close()
                if len(fold_preds) > 0 and os.path.exists(dca_png):
                    img = plt.imread(dca_png)
                    plt.figure(figsize=(6,5), dpi=200); plt.imshow(img); plt.axis('off'); pdf.savefig(bbox_inches='tight'); plt.close()
        except Exception:
            pass
    # ---- SHAP interpretability on full-data model (optional but enabled here) ----
    if TORCH_OK:
        try:
            model_full, D_full = dietnet_fit_full_model(
                X_fna.astype(np.float32), y_fna.astype(int), hidden=parse_hidden(args.hidden),
                epochs=args.epochs, lr=args.lr, w_l1=args.w_l1, seed=args.seed,
                device=pick_device(args.device)
            )
            make_shap_plots_torch(model_full, D_full, X_fna.astype(np.float32), feat_names, args.outdir, background_size=100, sample_force_index=0)
            print("[SHAP] Saved shap_summary.png, shap_beeswarm.png, shap_force.html")
        except Exception as e:
            print(f"[SHAP] Skipped due to error: {e}")

    # ---- Optional: calibration assessment
    if args.calibration and TORCH_OK:
        brier, calib_df = assess_calibration_cv(
            X_fna, y_fna, args.kfold,
            parse_hidden(args.hidden), args.epochs, args.lr, args.w_l1, args.seed,
            groups=groups_fna
        )
        with open(os.path.join(args.outdir, "fna_brier.txt"), "w") as f:
            f.write(f"Brier score: {brier:.4f}\n")
        calib_df.to_csv(os.path.join(args.outdir, "fna_calibration_curve.csv"), index=False)
        print("[Calib] Saved fna_brier.txt and fna_calibration_curve.csv")

    # ---- Optional: permutation test
    if args.perm_runs > 0 and TORCH_OK:
        auc_null = permutation_test_auc(
            X_fna, y_fna, args.kfold,
            parse_hidden(args.hidden), args.epochs, args.lr, args.w_l1, args.seed,
            groups=groups_fna, n_perm=args.perm_runs,
            device=pick_device(args.device), time_profile=args.time_profile
        )
        pd.DataFrame({"auc_null": auc_null}).to_csv(
            os.path.join(args.outdir, "fna_permutation_auc.csv"), index=False
        )
        # p-value = fraction of null >= observed mean (read from summary)
        try:
            with open(os.path.join(args.outdir, "fna_cv_summary.txt")) as f:
                lines = f.readlines()
            auc_line = [l for l in lines if l.startswith("AUC:")][0]
            auc_obs = float(auc_line.split()[1])
            pval = float((auc_null >= auc_obs).mean())
        except Exception:
            pval = np.nan
        with open(os.path.join(args.outdir, "fna_permutation_pvalue.txt"), "w") as f:
            f.write(f"p-value (AUC): {pval}\n")
        print("[Perm] Saved fna_permutation_auc.csv and fna_permutation_pvalue.txt")

    # ---- Optional: hyperparameter grid search
    if args.grid_hidden or args.grid_w_l1 or args.grid_epochs:
        grid_hidden = [s for s in (args.grid_hidden.split(';') if args.grid_hidden else [args.hidden])]
        grid_wl1 = [float(x) for x in (args.grid_w_l1.split(',') if args.grid_w_l1 else [args.w_l1])]
        grid_epochs = [int(x) for x in (args.grid_epochs.split(',') if args.grid_epochs else [args.epochs])]
        grid_df = hparam_grid_search(X_fna, y_fna, groups_fna, args.kfold, args.seed, grid_hidden, grid_wl1, grid_epochs, args.lr,
                                     device=pick_device(args.device), time_profile=args.time_profile)
        grid_df.to_csv(os.path.join(args.outdir, "fna_hparam_grid.csv"), index=False)
        print("[Grid] Saved fna_hparam_grid.csv")

    # ---- Optional: stability selection (L1-logit)
    if args.stability_B > 0:
        X_df = pd.DataFrame(X_fna, columns=feat_names)
        y_sr = pd.Series(y_fna, name="y")
        Cs = tuple(float(x) for x in args.stability_Cs.split(',') if x.strip())
        stab = stability_selection_logit(X_df, y_sr, B=args.stability_B, subsample=args.stability_subsample, Cs=Cs)
        stab.to_csv(os.path.join(args.outdir, "fna_stability_selection.csv"), index=False)
        print("[Stability] Saved fna_stability_selection.csv")

    # ---- Part 1b: External cohort application (train full FNA, predict external FNA)
    if args.external_csv is not None and TORCH_OK:
        df_ext = pd.read_csv(args.external_csv)
        if "sample" not in df_ext.columns:
            sys.exit("[ERROR] external_csv must contain a 'sample' column.")
        df_ext["sample"] = df_ext["sample"].astype(str)
        df_ext_fna = df_ext[df_ext["sample"].str.endswith("_FNA")].copy()
        if df_ext_fna.empty:
            print("[WARN] No _FNA rows in external cohort; skipping external apply.")
        else:
            # Align feature space: use columns present in training 'genomic_feat'
            feat_ext = df_ext_fna.reindex(columns=genomic_feat.columns, fill_value=np.nan)
            feat_ext = feat_ext.fillna(0).astype(float)

            # Fit on ALL internal FNA
            pred_fn, Ue_full = dietnet_fit_full(X_fna, y_fna, hidden=parse_hidden(args.hidden),
                                                epochs=args.epochs, lr=args.lr, w_l1=args.w_l1, seed=args.seed,
                                                device=pick_device(args.device))

            # (Optional) batched prediction for large cohorts
            def predict_proba_batched(Xnew: np.ndarray, batch_size: int = 16384) -> np.ndarray:
                probs_all = []
                for i in range(0, Xnew.shape[0], batch_size):
                    xb = Xnew[i:i+batch_size]
                    pb = pred_fn(xb)
                    probs_all.append(pb)
                return np.concatenate(probs_all, axis=0)

            X_ext = feat_ext.values.astype("float32", copy=False)
            probs = predict_proba_batched(X_ext, batch_size=16384)

            out = df_ext_fna[["sample"]].copy()
            out["prob_poor"] = probs
            # If external has label, compute quick metrics
            metrics_txt = ""
            if args.target in df_ext_fna.columns:
                y_ext = df_ext_fna[args.target].astype(int).values
                try:
                    auc_ext = roc_auc_score(y_ext, probs)
                except Exception:
                    auc_ext = np.nan
                y_pred = (probs >= 0.5).astype(int)
                acc_ext = accuracy_score(y_ext, y_pred)
                cm = confusion_matrix(y_ext, y_pred, labels=[0,1])
                metrics_txt = f"AUC_ext={auc_ext:.3f}\nACC_ext={acc_ext:.3f}\nConfusion:\n{cm}\n"
                with open(os.path.join(args.outdir, "external_metrics.txt"), "w") as f:
                    f.write(metrics_txt)
            out.to_csv(os.path.join(args.outdir, "external_predictions.csv"), index=False)
            print("[Part1b] Saved external_predictions.csv" + (" and external_metrics.txt" if metrics_txt else ""))

    # ---- Part 2: Delta features (mechanism)
    try:
        delta_df, y_pat = build_delta_matrix(
            df, target_col=args.target, drop_cols=drop_cols,
            seq_method_col=(args.platform_col if args.platform_col else 'seq_method'),
            pcga_set=PCGA_V3_SET, force_pcga_mask=args.restrict_to_pcga
        )
        # Optional collapse of Δ features to gene-level
        if args.collapse_gene_abn:
            before = delta_df.shape[1]
            delta_df = collapse_delta_by_gene(delta_df)
            after = delta_df.shape[1]
            try:
                print(f"[Collapse] Δ features collapsed to gene-level: {before} -> {after}")
            except Exception:
                pass
        uni = fisher_scan(delta_df, y_pat) if SCIPY_OK else pd.DataFrame()
        if not uni.empty:
            uni.to_csv(os.path.join(args.outdir, "delta_univariate.csv"), index=False)
        top = l1_multivariate(delta_df, y_pat, C=1.0)
        top.to_csv(os.path.join(args.outdir, "delta_multivariate_top.csv"), index=False)
        print("[Part2] Saved delta_univariate.csv and delta_multivariate_top.csv")
    except SystemExit as e:
        print(str(e))
        print("[Part2] Skipped delta binary analysis.")

    # ---- Part 2b: Tumor GSVA mechanism (optional; continuous variables)
    if args.gsva_tumor_tsv is not None:
        gsva_df = load_gsva_tumor(args.gsva_tumor_tsv)
        try:
            uni_gsva, top_gsva = tumor_gsva_mechanism(gsva_df, df, target_col=args.target)
            uni_gsva.to_csv(os.path.join(args.outdir, "tumor_gsva_univariate.csv"), index=False)
            top_gsva.to_csv(os.path.join(args.outdir, "tumor_gsva_multivariate_top.csv"), index=False)
            print("[Part2b] Saved tumor_gsva_univariate.csv and tumor_gsva_multivariate_top.csv")
        except SystemExit as e:
            print(str(e))

    print("All done.")


# ---------------------------
# OOF + comparison/pathway/volcano helpers
# ---------------------------

def oof_from_cv(X, y, kfold, hidden, epochs, lr, w_l1, seed, groups=None, y_strat=None, device=None):
    fold_preds, _ = collect_cv_predictions(X, y, kfold, hidden, epochs, lr, w_l1, seed, groups, y_strat, device)
    if len(fold_preds)==0:
        return np.array([]), np.array([])
    y_all = np.concatenate([it[0] for it in fold_preds])
    p_all = np.concatenate([it[1] for it in fold_preds])
    return y_all, p_all

def summarize_oof(y_all: np.ndarray, p_all: np.ndarray):
    out = {}
    if y_all.size==0:
        return dict(auc=np.nan, brier=np.nan, nb_avg=np.nan)
    try:
        out['auc'] = roc_auc_score(y_all, p_all)
    except Exception:
        out['auc'] = np.nan
    try:
        out['brier'] = brier_score_loss(y_all, p_all)
    except Exception:
        out['brier'] = np.nan
    thr, nb, nb_all = decision_curve(y_all, p_all, thresholds=np.linspace(0.1,0.5,41))
    # 平均ネットベネフィット（treat-all を差し引いた純増も参考に保存）
    out['nb_avg'] = float(np.nanmean(nb))
    out['nb_all_avg'] = float(np.nanmean(nb_all))
    out['nb_gain_avg'] = float(out['nb_avg'] - out['nb_all_avg'])
    return out

def make_volcano_from_univariate(csv_path: str, outdir: str, top_labels: int = 15):
    """Create volcano plot from fna_univariate.csv (or pathway_univariate.csv).
    x-axis priority: log2(OR) > standardized (median1-median0) > coef
    y-axis: -log10(q_bh) if present else -log10(p)
    Always logs why it skipped if it cannot produce a file.
    """
    try:
        dfu = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[Plot] Volcano skipped: cannot read {csv_path}: {e}")
        return
    if dfu.empty:
        print("[Plot] Volcano skipped: empty table")
        return

    df = dfu.copy()
    # y-axis: prefer q_bh over p
    if 'q_bh' in df.columns:
        q = pd.to_numeric(df['q_bh'], errors='coerce')
    elif 'p' in df.columns:
        q = pd.to_numeric(df['p'], errors='coerce')
    else:
        print("[Plot] Volcano skipped: neither q_bh nor p column present")
        return
    yv = -np.log10(q.clip(lower=1e-300))

    # x-axis candidates
    x = None
    if 'OR' in df.columns:
        OR = pd.to_numeric(df['OR'], errors='coerce')
        OR = OR.where(OR > 0, np.nan)
        with np.errstate(divide='ignore', invalid='ignore'):
            x_or = np.log2(OR)
        if x_or.notna().sum() >= max(10, int(0.2 * len(df))):
            x = x_or
    if x is None and {'median0','median1'}.issubset(df.columns):
        m0 = pd.to_numeric(df['median0'], errors='coerce')
        m1 = pd.to_numeric(df['median1'], errors='coerce')
        diff = m1 - m0
        scale = np.nanstd(diff.values)
        if not np.isfinite(scale) or scale == 0:
            scale = 1.0
        x = (diff / scale).replace([np.inf, -np.inf], np.nan)
    if x is None and 'coef' in df.columns:
        x = pd.to_numeric(df['coef'], errors='coerce')
    if x is None or x.notna().sum() == 0:
        print("[Plot] Volcano skipped: no usable x-axis (no OR/medians/coef)")
        return

    plt.figure(figsize=(7,6), dpi=200)
    plt.scatter(x, yv, alpha=0.6)
    try:
        plt.axvline(0, color='grey', linestyle='--', linewidth=1)
    except Exception:
        pass
    plt.axhline(-np.log10(0.1), color='grey', linestyle=':', linewidth=1)
    plt.xlabel('Effect (log2(OR) or proxy)')
    plt.ylabel('-log10(FDR q or p)')
    plt.title('Volcano plot: genetic feature vs CAP')

    # label top points by smallest q/p
    try:
        lab = df.copy()
        lab['_qplot'] = q
        lab = lab.sort_values('_qplot').head(top_labels)
        name_col = 'feature' if 'feature' in lab.columns else ('pathway' if 'pathway' in lab.columns else None)
        if name_col is not None:
            for _, r in lab.iterrows():
                xv = np.nan
                if 'OR' in df.columns and pd.notna(r.get('OR', np.nan)) and r.get('OR', np.nan) > 0:
                    xv = np.log2(r['OR'])
                elif {'median0','median1'}.issubset(df.columns):
                    base_sd = np.nanstd((df['median1']-df['median0']).values)
                    base_sd = 1.0 if (not np.isfinite(base_sd) or base_sd == 0) else base_sd
                    xv = (r.get('median1', np.nan) - r.get('median0', np.nan)) / base_sd
                elif 'coef' in df.columns:
                    xv = r.get('coef', np.nan)
                ylab = -np.log10(r.get('q_bh', r.get('p', 1.0)))
                if np.isfinite(xv) and np.isfinite(ylab):
                    plt.text(xv, ylab, str(r[name_col]), fontsize=8)
    except Exception:
        pass

    plt.tight_layout()
    out_png = os.path.join(outdir, 'fna_volcano.png')
    out_pdf = os.path.join(outdir, 'fna_volcano.pdf')
    try:
        plt.savefig(out_png, bbox_inches='tight', dpi=300)
        plt.savefig(out_pdf, bbox_inches='tight')
        print('[Plot] Saved fna_volcano.png')
    except Exception as e:
        print(f"[Plot] Volcano save failed: {e}")
    finally:
        plt.close()

def build_pathway_composites(X_df: pd.DataFrame, mapping_csv: str, mode: str = 'any') -> pd.DataFrame:
    """
    Build pathway-level composites from feature matrix X_df using a mapping CSV.
    Supported mapping CSV formats:
      1) columns: feature,pathway  (direct feature->pathway mapping)
      2) columns: Pathway,Gene     (gene->pathway; we will collapse multiple feature columns
                                    for the same gene, e.g., TP53_mut/amp/del/loh, into a single
                                    gene-level abnormality indicator to avoid double-counting.)

    Aggregation:
      - First collapse to gene-level abnormality (any abnormality across that gene's columns)
      - Then aggregate per pathway over its member genes:
          mode=='any':  any gene abnormal -> 1 else 0
          mode=='count': number of abnormal genes (float)
          mode=='mean': fraction of abnormal genes (mean of 0/1)
    """
    mapdf = pd.read_csv(mapping_csv)
    cols_lower = {c.lower(): c for c in mapdf.columns}

    # Normalize column names if user provides 'Pathway,Gene'
    if 'feature' in cols_lower and 'pathway' in cols_lower:
        # Direct feature->pathway mapping path
        feat_col = cols_lower['feature']
        pth_col  = cols_lower['pathway']
        m = mapdf[[feat_col, pth_col]].rename(columns={feat_col: 'feature', pth_col: 'pathway'})
        # keep only features that exist in X_df
        m = m[m['feature'].isin(X_df.columns)].copy()
        if m.empty:
            raise SystemExit('[pathway] No features in mapping matched model features')
        out = {}
        for pth, sub in m.groupby('pathway'):
            cols = [c for c in sub['feature'].tolist() if c in X_df.columns]
            if not cols:
                continue
            subX = X_df[cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).values
            # Treat >0 as abnormal for safety across binary/continuous encodings
            gene_abn = (subX > 0).any(axis=1).astype(int)
            # With direct feature mapping we already have per-feature; interpret them as genes
            if mode == 'any':
                comp = gene_abn.astype(int)  # any of mapped features abnormal
            elif mode == 'count':
                comp = (subX > 0).sum(axis=1).astype(float)  # number of abnormal mapped features
            else:  # mean
                comp = (subX > 0).mean(axis=1)
            out[pth] = comp
        if not out:
            raise SystemExit('[pathway] Mapping produced no composites; check mapping CSV')
        return pd.DataFrame(out, index=X_df.index)

    # Gene->pathway mapping path (columns like 'Pathway','Gene')
    if ('pathway' in cols_lower or 'Pathway' in mapdf.columns) and ('gene' in cols_lower or 'Gene' in mapdf.columns):
        pth_col = cols_lower.get('pathway', 'Pathway')
        g_col   = cols_lower.get('gene', 'Gene')
        m = mapdf[[pth_col, g_col]].rename(columns={pth_col: 'pathway', g_col: 'gene'})

        # Also create a fast lookup of all columns lowercased
        cols = np.array(X_df.columns)
        cols_lower_arr = np.char.lower(cols.astype(str))

        # For each gene, find matching columns and collapse to a single 0/1 abnormality
        gene_to_abn = {}
        for gene in pd.unique(m['gene'].astype(str)):
            gl = gene.lower()
            # candidate columns that begin with gene name (strict prefix match), allowing suffixes
            # e.g., TP53, TP53_mut, TP53-amp, TP53_del, TP53_loh, TP53_cnvloss, etc.
            starts = np.char.startswith(cols_lower_arr, gl)
            cand_cols = cols[starts].tolist()
            if not cand_cols:
                continue
            subX = X_df[cand_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
            # Abnormal if value > 0 in ANY of the gene's columns (covers mut/amp/del/loh/cnv*)
            abn = (subX.values > 0).any(axis=1).astype(int)
            gene_to_abn[gene] = abn
        if not gene_to_abn:
            raise SystemExit('[pathway] No gene columns matched X_df; check naming (expect gene[_suffix])')
        G = pd.DataFrame(gene_to_abn, index=X_df.index)  # columns=genes, values in {0,1}

        # Aggregate to pathways without double-counting within a gene (already collapsed)
        out = {}
        for pth, sub in m.groupby('pathway'):
            genes = [g for g in sub['gene'].astype(str) if g in G.columns]
            if not genes:
                continue
            subG = G[genes]
            if mode == 'any':
                comp = (subG.values > 0).any(axis=1).astype(int)
            elif mode == 'count':
                comp = (subG.values > 0).sum(axis=1).astype(float)  # number of abnormal genes in the pathway
            else:  # mean
                comp = (subG.values > 0).mean(axis=1)
            out[pth] = comp
        if not out:
            raise SystemExit('[pathway] Gene mapping produced no composites; check mapping CSV contents')
        return pd.DataFrame(out, index=X_df.index)

    raise SystemExit('[pathway] CSV must contain either columns: feature,pathway OR Pathway,Gene')


if __name__ == "__main__":
    main()
