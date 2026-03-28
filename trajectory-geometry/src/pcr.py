"""Probability Cloud Regression (PCR) utilities.

Two closely related denoising styles appear across the archived experiments:

1. A Mahalanobis/CloudRegressor formulation used in EXP-19B and EXP-21A.
2. A shrinkage-to-mean formulation used in later gap-filling and notebook work.

Both are exposed here. The notebook workflow defaults to the shrinkage form
because it is easy to explain and directly matches the paper-facing summary:

    z_hat = x_bar + (sigma_z^2 / (sigma_z^2 + sigma_i^2)) * (x_i - x_bar)

The CloudRegressor implementation is kept for fidelity with the archived
cross-architecture PCR reanalyses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class PCRSummary:
    """Summary statistics for a shrinkage denoising pass."""

    metric: str
    group_mean: float
    latent_variance: float
    noise_variance: float
    mean_reliability: float


class CloudRegressor:
    """Errors-in-variables regression with per-observation uncertainty."""

    def __init__(self) -> None:
        self.m: Optional[float] = None
        self.c: Optional[float] = None
        self.inferred_x: Optional[np.ndarray] = None

    @staticmethod
    def _objective(
        params: np.ndarray,
        x_obs: np.ndarray,
        y_obs: np.ndarray,
        sig_x: np.ndarray,
        sig_y: np.ndarray,
    ) -> float:
        m, c = params
        w_x = 1.0 / (sig_x**2)
        w_y = 1.0 / (sig_y**2)
        x_hat = (w_x * x_obs + w_y * m * (y_obs - c)) / (w_x + w_y * (m**2))
        y_hat = m * x_hat + c
        dist_sq = w_x * (x_obs - x_hat) ** 2 + w_y * (y_obs - y_hat) ** 2
        return float(np.sum(dist_sq))

    def fit(
        self,
        x_obs: np.ndarray,
        y_obs: np.ndarray,
        sig_x: np.ndarray,
        sig_y: np.ndarray,
    ) -> "CloudRegressor":
        sig_x = np.maximum(np.asarray(sig_x, dtype=float), 1e-6)
        sig_y = np.maximum(np.asarray(sig_y, dtype=float), 1e-6)
        design = np.vstack([x_obs, np.ones(len(x_obs))]).T
        m0, c0 = np.linalg.lstsq(design, y_obs, rcond=None)[0]
        result = minimize(
            self._objective,
            np.array([m0, c0], dtype=float),
            args=(x_obs, y_obs, sig_x, sig_y),
            method="Nelder-Mead",
            options={"maxiter": 10000, "xatol": 1e-6, "fatol": 1e-6},
        )
        self.m, self.c = result.x
        w_x = 1.0 / (sig_x**2)
        w_y = 1.0 / (sig_y**2)
        self.inferred_x = (w_x * x_obs + w_y * self.m * (y_obs - self.c)) / (
            w_x + w_y * (self.m**2)
        )
        return self

    def get_denoised_x(self) -> np.ndarray:
        if self.inferred_x is None:
            raise RuntimeError("CloudRegressor must be fitted before reading denoised values.")
        return self.inferred_x


def estimate_sigma_strategy_d(
    df: pd.DataFrame,
    metrics: Sequence[str],
    fraction: float = 0.25,
) -> Dict[str, float]:
    """Strategy D from the archive: sigma = fraction × column standard deviation."""

    return {metric: float(df[metric].std() * fraction) for metric in metrics if metric in df}


def estimate_sigma_from_cross_layer_variance(
    df: pd.DataFrame,
    metrics: Sequence[str],
    group_cols: Sequence[str] = ("problem_id", "condition"),
) -> Dict[str, pd.Series]:
    """Estimate per-row sigma from cross-layer within-trajectory variance.

    Returns a mapping from metric name to a row-aligned series. Each row receives
    the standard deviation of that metric across the corresponding trajectory.
    """

    sigmas: Dict[str, pd.Series] = {}
    grouped = df.groupby(list(group_cols), dropna=False)
    for metric in metrics:
        if metric not in df.columns:
            continue
        sigma_series = grouped[metric].transform("std").fillna(0.0)
        sigmas[metric] = sigma_series
    return sigmas


def shrinkage_denoise_series(
    values: pd.Series,
    sigma: pd.Series | float,
    group_mean: Optional[float] = None,
) -> tuple[pd.Series, PCRSummary]:
    """Denoise a single metric with the shrinkage PCR update."""

    raw = values.astype(float)
    sigma_values = sigma if isinstance(sigma, pd.Series) else pd.Series(float(sigma), index=raw.index)
    mean_val = float(raw.mean()) if group_mean is None else float(group_mean)
    noise_variance = np.square(sigma_values.astype(float))
    observed_variance = float(np.nanvar(raw.to_numpy(dtype=float)))
    mean_noise = float(np.nanmean(noise_variance.to_numpy(dtype=float)))
    latent_variance = max(observed_variance - mean_noise, 0.0)
    reliability = latent_variance / (latent_variance + noise_variance.replace(0, np.nan))
    reliability = reliability.fillna(0.0).clip(lower=0.0, upper=1.0)
    denoised = mean_val + reliability * (raw - mean_val)
    summary = PCRSummary(
        metric=raw.name or "metric",
        group_mean=mean_val,
        latent_variance=latent_variance,
        noise_variance=mean_noise,
        mean_reliability=float(reliability.mean()),
    )
    return denoised, summary


def shrinkage_denoise_dataframe(
    df: pd.DataFrame,
    metrics: Sequence[str],
    sigmas: Dict[str, pd.Series | float],
    group_col: Optional[str] = None,
) -> tuple[pd.DataFrame, list[PCRSummary]]:
    """Apply shrinkage PCR to a dataframe of metrics."""

    denoised = df.copy()
    summaries: list[PCRSummary] = []
    for metric in metrics:
        if metric not in denoised.columns or metric not in sigmas:
            continue
        if group_col and group_col in denoised.columns:
            pieces = []
            for _, group_df in denoised.groupby(group_col, dropna=False):
                sigma = sigmas[metric]
                sigma_piece = sigma.loc[group_df.index] if isinstance(sigma, pd.Series) else sigma
                cleaned, summary = shrinkage_denoise_series(
                    group_df[metric],
                    sigma=sigma_piece,
                    group_mean=float(group_df[metric].mean()),
                )
                pieces.append(cleaned)
                summaries.append(summary)
            denoised.loc[pd.concat(pieces).index, metric] = pd.concat(pieces).sort_index()
        else:
            cleaned, summary = shrinkage_denoise_series(denoised[metric], sigmas[metric])
            denoised[metric] = cleaned
            summaries.append(summary)
    return denoised, summaries


def pcr_denoise_dataframe(
    df: pd.DataFrame,
    metrics: Sequence[str],
    sigmas: Dict[str, float],
    anchor_col: str = "problem_id",
) -> pd.DataFrame:
    """Apply CloudRegressor denoising to selected columns."""

    denoised = df.copy()
    unique_vals = sorted(denoised[anchor_col].dropna().unique())
    if not unique_vals:
        return denoised

    id_map = {value: idx for idx, value in enumerate(unique_vals)}
    anchor = denoised[anchor_col].map(id_map).astype(float).to_numpy()
    anchor_max = np.nanmax(anchor) if len(anchor) else 1.0
    if anchor_max > 0:
        anchor = anchor / anchor_max
    sig_y = np.ones(len(denoised), dtype=float) * 0.05

    for metric in metrics:
        if metric not in denoised.columns:
            continue
        sigma = float(sigmas.get(metric, np.nan))
        raw = denoised[metric].astype(float).to_numpy()
        if np.isnan(sigma) or sigma <= 0 or np.nanstd(raw) < 1e-12:
            continue
        nan_mask = np.isnan(raw)
        fill_value = float(np.nanmean(raw)) if not np.all(nan_mask) else 0.0
        x_obs = np.where(nan_mask, fill_value, raw)
        sig_x = np.ones(len(x_obs), dtype=float) * max(sigma, 1e-4)
        reg = CloudRegressor().fit(x_obs, anchor, sig_x, sig_y)
        denoised_vals = reg.get_denoised_x()
        denoised_vals[nan_mask] = np.nan
        denoised[metric] = denoised_vals

    return denoised
