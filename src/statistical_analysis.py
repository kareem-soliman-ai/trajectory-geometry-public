"""Statistical analysis helpers for trajectory-geometry experiments."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from statsmodels.formula.api import ols

from .pcr import (
    estimate_sigma_from_cross_layer_variance,
    pcr_denoise_dataframe,
    shrinkage_denoise_dataframe,
)


def cohens_d(group_a: Sequence[float], group_b: Sequence[float]) -> float:
    """Compute Cohen's d with pooled standard deviation."""

    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = np.sqrt(
        (((len(a) - 1) * np.var(a, ddof=1)) + ((len(b) - 1) * np.var(b, ddof=1)))
        / (len(a) + len(b) - 2)
    )
    if pooled <= 1e-12:
        return np.nan
    return float((np.mean(a) - np.mean(b)) / pooled)


def permutation_test_mean_difference(
    group_a: Sequence[float],
    group_b: Sequence[float],
    n_perm: int = 10_000,
    seed: int = 42,
) -> float:
    """Two-sided permutation test on the mean difference."""

    rng = np.random.default_rng(seed)
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    observed = float(np.mean(a) - np.mean(b))
    pooled = np.concatenate([a, b]).copy()
    extreme = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        diff = np.mean(pooled[: len(a)]) - np.mean(pooled[len(a) :])
        if abs(diff) >= abs(observed):
            extreme += 1
    return float((extreme + 1) / (n_perm + 1))


def two_way_anova(df: pd.DataFrame, metric: str) -> Optional[Dict[str, float]]:
    """Run a 2-way ANOVA with Regime × Correctness factors."""

    subset = df.dropna(subset=[metric, "condition", "correct"]).copy()
    if subset.empty:
        return None
    formula = f'Q("{metric}") ~ C(condition) * C(correct)'
    model = ols(formula, data=subset).fit()
    aov = sm.stats.anova_lm(model, typ=2)
    ss_total = float(aov["sum_sq"].sum())
    if ss_total <= 1e-12:
        return None
    ss_regime = float(aov.loc["C(condition)", "sum_sq"])
    ss_quality = float(aov.loc["C(correct)", "sum_sq"])
    ss_interaction = float(aov.loc["C(condition):C(correct)", "sum_sq"])
    ss_residual = float(aov.loc["Residual", "sum_sq"])
    explained = ss_regime + ss_quality + ss_interaction
    return {
        "SS_regime": ss_regime,
        "SS_quality": ss_quality,
        "SS_interaction": ss_interaction,
        "SS_residual": ss_residual,
        "SS_total": ss_total,
        "eta_sq_regime": ss_regime / ss_total,
        "eta_sq_quality": ss_quality / ss_total,
        "eta_sq_interaction": ss_interaction / ss_total,
        "partial_eta_sq_regime": ss_regime / (ss_regime + ss_residual),
        "partial_eta_sq_quality": ss_quality / (ss_quality + ss_residual),
        "partial_eta_sq_interaction": ss_interaction / (ss_interaction + ss_residual),
        "proportion_regime_of_explained": (ss_regime / explained) if explained > 0 else np.nan,
        "f_regime": float(aov.loc["C(condition)", "F"]),
        "p_regime": float(aov.loc["C(condition)", "PR(>F)"]),
        "f_quality": float(aov.loc["C(correct)", "F"]),
        "p_quality": float(aov.loc["C(correct)", "PR(>F)"]),
        "f_interaction": float(aov.loc["C(condition):C(correct)", "F"]),
        "p_interaction": float(aov.loc["C(condition):C(correct)", "PR(>F)"]),
    }


def stratified_logistic_auc(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str = "correct",
    n_splits: int = 5,
    c_value: float = 0.1,
    max_iter: int = 1000,
    random_state: int = 42,
) -> tuple[float, float]:
    """Run stratified cross-validated logistic regression."""

    work = df.dropna(subset=list(feature_cols) + [target_col]).copy()
    if work.empty or work[target_col].nunique() < 2:
        return np.nan, np.nan

    X = work.loc[:, list(feature_cols)].astype(float).to_numpy()
    y = work[target_col].astype(int).to_numpy()

    class_counts = np.bincount(y)
    viable = class_counts[class_counts > 0]
    n_splits = min(n_splits, int(viable.min()) if len(viable) else 0)
    if n_splits < 2:
        return np.nan, np.nan

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scaler = StandardScaler()
    aucs: list[float] = []

    for train_idx, test_idx in splitter.split(X, y):
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        y_train, y_test = y[train_idx], y[test_idx]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue
        clf = LogisticRegression(C=c_value, max_iter=max_iter, solver="lbfgs")
        clf.fit(X_train, y_train)
        prob = clf.predict_proba(X_test)[:, 1]
        aucs.append(roc_auc_score(y_test, prob))

    if not aucs:
        return np.nan, np.nan
    return float(np.mean(aucs)), float(np.std(aucs))


def pcr_corrected_auc(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str = "correct",
    denoiser: str = "shrinkage",
    anchor_col: str = "problem_id",
) -> dict[str, float]:
    """Compute raw and PCR-corrected AUCs from the same dataframe."""

    raw_auc, raw_std = stratified_logistic_auc(df, feature_cols, target_col=target_col)
    sigmas = estimate_sigma_from_cross_layer_variance(df, feature_cols)
    if denoiser == "cloud":
        scalar_sigmas = {
            metric: float(series.mean()) for metric, series in sigmas.items() if len(series) > 0
        }
        denoised = pcr_denoise_dataframe(df, feature_cols, scalar_sigmas, anchor_col=anchor_col)
    else:
        denoised, _ = shrinkage_denoise_dataframe(df, feature_cols, sigmas)
    pcr_auc, pcr_std = stratified_logistic_auc(denoised, feature_cols, target_col=target_col)
    return {
        "raw_auc": raw_auc,
        "raw_std": raw_std,
        "pcr_auc": pcr_auc,
        "pcr_std": pcr_std,
        "gain": pcr_auc - raw_auc if pd.notna(raw_auc) and pd.notna(pcr_auc) else np.nan,
    }


def effect_grid(
    df: pd.DataFrame,
    metrics: Sequence[str],
    group_a: str,
    group_b: str,
    layer_col: str = "layer",
    group_col: str = "group",
) -> pd.DataFrame:
    """Return layer-resolved Cohen's d values for a metric list."""

    results = []
    for metric in metrics:
        for layer in sorted(df[layer_col].dropna().unique()):
            a = df[(df[group_col] == group_a) & (df[layer_col] == layer)][metric]
            b = df[(df[group_col] == group_b) & (df[layer_col] == layer)][metric]
            results.append(
                {
                    "metric": metric,
                    "layer": int(layer),
                    "group_a": group_a,
                    "group_b": group_b,
                    "cohen_d": cohens_d(a, b),
                    "perm_p": permutation_test_mean_difference(a, b),
                }
            )
    return pd.DataFrame(results)
