"""Trajectory metric computation for hidden-state geometry analyses.

This module consolidates the broad metric families used across the trajectory
geometry experiments into one documented public API. The implementation is
adapted from the EXP-18 metric suite, the EXP-16 recomputation utilities, and
later commitment-focused analysis scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
from scipy.signal import welch
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy, linregress


def _lz_complexity(binary_sequence: Iterable[int]) -> int:
    """Simple Lempel-Ziv complexity estimate for a binary sequence."""

    seq = list(binary_sequence)
    n = len(seq)
    if n == 0:
        return 0
    i, k, l = 1, 1, 1
    while i + k <= n:
        if seq[i + k - 1] == seq[l + k - 1]:
            k += 1
        else:
            l = 0
            while l < i:
                if seq[l : l + k] == seq[i : i + k]:
                    break
                l += 1
            if l == i:
                i += k
                l = 1
                k = 1
            else:
                k += 1
    return l


@dataclass
class MetricContext:
    """Optional contextual inputs for semantic or attractor-aware metrics."""

    truth_id: Optional[int] = None
    wrong_id: Optional[int] = None
    operand_ids: tuple[int, ...] = ()
    intermediate_id: Optional[int] = None
    centroids_by_layer: Optional[dict[int, np.ndarray]] = None


class TrajectoryMetrics:
    """Compute trajectory metrics on a single layer or full layer stack."""

    def __init__(
        self,
        unembedding_matrix: Optional[np.ndarray] = None,
        embedding_matrix: Optional[np.ndarray] = None,
        gram_matrix: Optional[np.ndarray] = None,
    ) -> None:
        self.unembedding_matrix = unembedding_matrix
        self.embedding_matrix = embedding_matrix
        self.gram_matrix = gram_matrix

    @staticmethod
    def _safe_norm(values: np.ndarray, axis=None) -> np.ndarray:
        norms = np.linalg.norm(values, axis=axis)
        return np.maximum(norms, 1e-12)

    @staticmethod
    def _deltas(h: np.ndarray) -> np.ndarray:
        return h[1:] - h[:-1]

    @staticmethod
    def radius_of_gyration(h: np.ndarray) -> float:
        centered = h - np.mean(h, axis=0)
        return float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))

    def kinematic_metrics(self, h: np.ndarray) -> Dict[str, float]:
        """Family 1: motion-style metrics over token displacements."""

        metrics: Dict[str, float] = {
            "speed": np.nan,
            "turn_angle": np.nan,
            "tortuosity": np.nan,
            "directional_consistency": np.nan,
            "stabilization_rate": np.nan,
            "vel_autocorr_lag1": np.nan,
            "vel_autocorr_lag2": np.nan,
            "vel_autocorr_lag4": np.nan,
            "vel_autocorr_lag8": np.nan,
            "dir_autocorr_lag1": np.nan,
            "dir_autocorr_lag2": np.nan,
            "dir_autocorr_lag4": np.nan,
            "dir_autocorr_lag8": np.nan,
        }
        if len(h) < 3:
            return metrics

        deltas = self._deltas(h)
        delta_norms = self._safe_norm(deltas, axis=1)
        metrics["speed"] = float(np.mean(delta_norms))

        numer = np.sum(deltas[:-1] * deltas[1:], axis=1)
        denom = delta_norms[:-1] * delta_norms[1:]
        metrics["turn_angle"] = float(np.mean(np.arccos(np.clip(numer / denom, -1, 1))))
        metrics["tortuosity"] = float(np.linalg.norm(h[-1] - h[0]) / (np.sum(delta_norms) + 1e-12))

        norm_deltas = deltas / delta_norms[:, None]
        metrics["directional_consistency"] = float(np.linalg.norm(np.mean(norm_deltas, axis=0)))
        if len(delta_norms) > 1:
            metrics["stabilization_rate"] = float(
                linregress(np.arange(len(delta_norms)), delta_norms).slope
            )

        for lag in (1, 2, 4, 8):
            if len(delta_norms) > lag:
                if np.std(delta_norms) > 0:
                    metrics[f"vel_autocorr_lag{lag}"] = float(
                        np.corrcoef(delta_norms[:-lag], delta_norms[lag:])[0, 1]
                    )
                metrics[f"dir_autocorr_lag{lag}"] = float(
                    np.mean(np.sum(norm_deltas[:-lag] * norm_deltas[lag:], axis=1))
                )
        return metrics

    def volumetric_metrics(self, h: np.ndarray) -> Dict[str, float]:
        """Family 2: space usage and dimensionality metrics."""

        metrics = {
            "radius_of_gyration": np.nan,
            "effective_dimension": np.nan,
            "gyration_anisotropy": np.nan,
            "drift_to_spread": np.nan,
        }
        if len(h) < 3:
            return metrics

        centered = h - np.mean(h, axis=0)
        metrics["radius_of_gyration"] = self.radius_of_gyration(h)
        deltas = self._deltas(h)
        centered_deltas = deltas - np.mean(deltas, axis=0)
        try:
            _, singular_values, _ = np.linalg.svd(centered_deltas, full_matrices=False)
            eigenvalues = singular_values**2
            metrics["effective_dimension"] = float(
                (np.sum(eigenvalues) ** 2) / (np.sum(eigenvalues**2) + 1e-12)
            )
        except np.linalg.LinAlgError:
            pass

        try:
            _, singular_values_h, _ = np.linalg.svd(centered, full_matrices=False)
            normed = singular_values_h / np.sum(singular_values_h)
            metrics["gyration_anisotropy"] = float(
                1 - entropy(normed) / np.log(max(min(len(h), h.shape[1]), 2))
            )
        except np.linalg.LinAlgError:
            pass

        metrics["drift_to_spread"] = float(
            np.linalg.norm(h[-1] - h[0]) / (metrics["radius_of_gyration"] + 1e-12)
        )
        return metrics

    def convergence_metrics(self, h: np.ndarray, window: int = 6, stride: int = 2) -> Dict[str, float]:
        """Family 3: approach-to-answer and commitment dynamics."""

        metrics = {
            "cosine_slope": np.nan,
            "distance_slope": np.nan,
            "early_late_ratio": np.nan,
            "time_to_commit": np.nan,
            "commitment_sharpness": np.nan,
            "cosine_to_late_window": np.nan,
            "cosine_to_running_mean": np.nan,
        }
        if len(h) < 5:
            return metrics

        final = h[-1]
        norm_final = np.linalg.norm(final) + 1e-12
        cosines = [
            np.dot(vec, final) / ((np.linalg.norm(vec) + 1e-12) * norm_final)
            for vec in h[:-1]
        ]
        distances = [np.linalg.norm(vec - final) for vec in h[:-1]]
        metrics["cosine_slope"] = float(linregress(np.arange(len(cosines)), cosines).slope)
        metrics["distance_slope"] = float(linregress(np.arange(len(distances)), distances).slope)

        step_norms = np.linalg.norm(self._deltas(h), axis=1)
        midpoint = len(step_norms) // 2
        if midpoint > 0:
            metrics["early_late_ratio"] = float(
                np.mean(step_norms[:midpoint]) / (np.mean(step_norms[midpoint:]) + 1e-12)
            )

        rg_values = []
        positions = []
        for start in range(0, len(h) - window + 1, stride):
            chunk = h[start : start + window]
            rg = self.radius_of_gyration(chunk)
            if not np.isnan(rg):
                rg_values.append(rg)
                positions.append(start + window // 2)
        if len(rg_values) >= 3:
            rg_values = np.asarray(rg_values, dtype=float)
            drops = rg_values[:-1] - rg_values[1:]
            max_drop_idx = int(np.argmax(drops))
            metrics["time_to_commit"] = float(positions[max_drop_idx])
            metrics["commitment_sharpness"] = float(np.max(drops))

        late_mean = np.mean(h[-min(4, len(h)) :], axis=0)
        late_norm = np.linalg.norm(late_mean) + 1e-12
        metrics["cosine_to_late_window"] = float(
            np.mean(
                [
                    np.dot(vec, late_mean) / ((np.linalg.norm(vec) + 1e-12) * late_norm)
                    for vec in h
                ]
            )
        )

        running_mean = np.cumsum(h, axis=0) / np.arange(1, len(h) + 1)[:, None]
        running_cosines = []
        for idx in range(2, len(h)):
            delta = h[idx] - h[idx - 1]
            reference = running_mean[idx - 1] - h[0]
            denom = (np.linalg.norm(delta) + 1e-12) * (np.linalg.norm(reference) + 1e-12)
            running_cosines.append(np.dot(delta, reference) / denom)
        if running_cosines:
            metrics["cosine_to_running_mean"] = float(np.mean(running_cosines))
        return metrics

    def diffusion_metrics(self, h: np.ndarray) -> Dict[str, float]:
        """Family 4: diffusion and spectral diagnostics."""

        metrics = {
            "msd_exponent": np.nan,
            "spectral_entropy": np.nan,
            "psd_slope": np.nan,
        }
        if len(h) >= 10:
            max_tau = max(2, len(h) // 2)
            taus = np.unique(np.logspace(0, np.log10(max_tau), num=min(10, max_tau)).astype(int))
            msds = []
            valid_taus = []
            for tau in taus:
                if tau <= 0 or tau >= len(h):
                    continue
                diffs = h[tau:] - h[:-tau]
                msd = np.mean(np.sum(diffs**2, axis=1))
                if msd > 0:
                    valid_taus.append(tau)
                    msds.append(msd)
            if len(msds) > 1:
                metrics["msd_exponent"] = float(
                    linregress(np.log(valid_taus), np.log(msds)).slope
                )

        norms = np.linalg.norm(self._deltas(h), axis=1) if len(h) > 1 else np.array([])
        if len(norms) >= 5:
            freqs, psd = welch(norms)
            normed_psd = psd / np.sum(psd)
            valid = (freqs > 0) & (psd > 0)
            metrics["spectral_entropy"] = float(entropy(normed_psd))
            if np.sum(valid) > 2:
                metrics["psd_slope"] = float(
                    linregress(np.log(freqs[valid]), np.log(psd[valid])).slope
                )
        return metrics

    def recurrence_metrics(self, h: np.ndarray) -> Dict[str, float]:
        """Family 5: recurrence quantification analysis metrics."""

        metrics = {
            "recurrence_rate": np.nan,
            "determinism": np.nan,
            "laminarity": np.nan,
            "trapping_time": np.nan,
            "diagonal_entropy": np.nan,
        }
        if len(h) < 5:
            return metrics

        distances = pdist(h)
        if len(distances) == 0:
            return metrics
        matrix = squareform(distances)
        epsilon = np.percentile(distances, 10)
        epsilon = epsilon if epsilon > 0 else 1e-9
        recurrence = (matrix < epsilon).astype(int)
        np.fill_diagonal(recurrence, 0)
        total_recurrence = np.sum(recurrence)
        metrics["recurrence_rate"] = float(total_recurrence / (len(h) ** 2))

        diagonal_runs = []
        vertical_runs = []
        for offset in range(1, len(h)):
            diag = np.diagonal(recurrence, offset)
            run = 0
            for value in diag:
                if value:
                    run += 1
                else:
                    if run >= 2:
                        diagonal_runs.append(run)
                    run = 0
            if run >= 2:
                diagonal_runs.append(run)

        for column_idx in range(len(h)):
            run = 0
            for value in recurrence[:, column_idx]:
                if value:
                    run += 1
                else:
                    if run >= 2:
                        vertical_runs.append(run)
                    run = 0
            if run >= 2:
                vertical_runs.append(run)

        if total_recurrence > 0:
            metrics["determinism"] = float(np.sum(diagonal_runs) / total_recurrence)
            metrics["laminarity"] = float(np.sum(vertical_runs) / total_recurrence)
        if vertical_runs:
            metrics["trapping_time"] = float(np.mean(vertical_runs))
        if diagonal_runs:
            metrics["diagonal_entropy"] = float(entropy(diagonal_runs))
        return metrics

    def landmark_metrics(self, h: np.ndarray, context: MetricContext) -> Dict[str, float]:
        """Family 6: semantic landmark metrics requiring unembedding vectors."""

        metrics = {
            "final_correct_logit": np.nan,
            "final_wrong_logit": np.nan,
            "logit_gap": np.nan,
            "operand_0_proximity": np.nan,
            "operand_1_proximity": np.nan,
            "operand_2_proximity": np.nan,
            "intermediate_proximity": np.nan,
            "landmark_crossing_order_entropy": np.nan,
        }
        if self.unembedding_matrix is None or context.truth_id is None:
            return metrics

        correct = self.unembedding_matrix[context.truth_id]
        correct_logits = h @ correct
        metrics["final_correct_logit"] = float(correct_logits[-1])

        if context.wrong_id is not None:
            wrong = self.unembedding_matrix[context.wrong_id]
            wrong_logits = h @ wrong
            metrics["final_wrong_logit"] = float(wrong_logits[-1])
            metrics["logit_gap"] = float(np.mean(correct_logits[-4:] - wrong_logits[-4:]))

        visit_order = []
        for idx in range(3):
            if idx < len(context.operand_ids):
                operand_vec = self.unembedding_matrix[context.operand_ids[idx]]
                prox = [
                    np.dot(vec, operand_vec)
                    / ((np.linalg.norm(vec) + 1e-12) * (np.linalg.norm(operand_vec) + 1e-12))
                    for vec in h
                ]
                metrics[f"operand_{idx}_proximity"] = float(np.max(prox))
                visit_order.append(int(np.argmax(prox)))
        if context.intermediate_id is not None:
            intermediate = self.unembedding_matrix[context.intermediate_id]
            prox = [
                np.dot(vec, intermediate)
                / ((np.linalg.norm(vec) + 1e-12) * (np.linalg.norm(intermediate) + 1e-12))
                for vec in h
            ]
            metrics["intermediate_proximity"] = float(np.max(prox))
        if visit_order:
            histogram = np.histogram(visit_order, bins=max(len(h), 2))[0]
            metrics["landmark_crossing_order_entropy"] = float(entropy(histogram + 1e-12))
        return metrics

    def attractor_metrics(self, h: np.ndarray, centroids: Optional[np.ndarray]) -> Dict[str, float]:
        """Family 7: success-attractor metrics."""

        metrics = {
            "mean_attractor_distance": np.nan,
            "attractor_divergence_slope": np.nan,
            "cosine_to_success_direction": np.nan,
            "local_expansion_rate": np.nan,
            "point_of_no_return_token": np.nan,
        }
        if centroids is None or len(h) == 0:
            return metrics

        limit = min(len(h), len(centroids))
        distances = np.linalg.norm(h[:limit] - centroids[:limit], axis=1)
        metrics["mean_attractor_distance"] = float(np.mean(distances))
        if limit > 2:
            metrics["attractor_divergence_slope"] = float(
                linregress(np.arange(limit), distances).slope
            )

        deltas = self._deltas(h)[: max(limit - 1, 0)]
        direction_to_centroid = centroids[:limit] - h[:limit]
        cosines = []
        for idx in range(len(deltas)):
            denom = (np.linalg.norm(deltas[idx]) + 1e-12) * (
                np.linalg.norm(direction_to_centroid[idx]) + 1e-12
            )
            cosines.append(np.dot(deltas[idx], direction_to_centroid[idx]) / denom)
        if cosines:
            metrics["cosine_to_success_direction"] = float(np.mean(cosines))

        step_norms = np.linalg.norm(self._deltas(h), axis=1)
        if len(step_norms) > 1:
            metrics["local_expansion_rate"] = float(
                np.mean(step_norms[1:] / (step_norms[:-1] + 1e-12))
            )

        mean_distance = float(np.mean(distances))
        candidates = np.where(distances > mean_distance * 2)[0]
        metrics["point_of_no_return_token"] = float(candidates[0] if len(candidates) else len(h))
        return metrics

    def embedding_stability_metrics(self, h: np.ndarray, context: MetricContext) -> Dict[str, float]:
        """Family 8: stability of embedding/unembedding relationships."""

        metrics = {
            "logit_consistency": np.nan,
            "landmark_pair_similarity": np.nan,
            "embed_unembed_alignment": np.nan,
        }
        if (
            self.unembedding_matrix is None
            or self.embedding_matrix is None
            or context.truth_id is None
        ):
            return metrics

        logits = h @ self.unembedding_matrix[context.truth_id]
        metrics["logit_consistency"] = float(np.std(logits) / (abs(np.mean(logits)) + 1e-12))

        if context.operand_ids:
            operand_id = context.operand_ids[0]
            operand_vec = self.unembedding_matrix[operand_id]
            truth_vec = self.unembedding_matrix[context.truth_id]
            metrics["landmark_pair_similarity"] = float(
                np.dot(operand_vec, truth_vec)
                / ((np.linalg.norm(operand_vec) + 1e-12) * (np.linalg.norm(truth_vec) + 1e-12))
            )

        metrics["embed_unembed_alignment"] = float(
            np.dot(self.embedding_matrix[context.truth_id], self.unembedding_matrix[context.truth_id])
            / (
                (np.linalg.norm(self.embedding_matrix[context.truth_id]) + 1e-12)
                * (np.linalg.norm(self.unembedding_matrix[context.truth_id]) + 1e-12)
            )
        )
        return metrics

    def information_metrics(self, h: np.ndarray) -> Dict[str, float]:
        """Family 9: information-theoretic trajectory metrics."""

        metrics = {
            "step_surprisal": np.nan,
            "trajectory_entropy_rate": np.nan,
            "info_gain_proxy": np.nan,
        }
        if len(h) < 3:
            return metrics

        deltas = self._deltas(h)
        norms = self._safe_norm(deltas, axis=1)
        normed = deltas / norms[:, None]
        if len(normed) > 1:
            surprisal = 1 - np.sum(normed[1:] * normed[:-1], axis=1)
            metrics["step_surprisal"] = float(np.mean(surprisal))

        if len(norms) > 1:
            disc = (np.diff(norms) > 0).astype(int)
            metrics["trajectory_entropy_rate"] = float(_lz_complexity(disc) / (len(disc) + 1e-12))

        diff = (h[-1] - h[0])[None, :]
        if self.gram_matrix is not None:
            metrics["info_gain_proxy"] = float(
                np.sqrt(np.maximum(0, (diff @ self.gram_matrix @ diff.T)[0, 0]))
            )
        elif self.unembedding_matrix is not None:
            metrics["info_gain_proxy"] = float(
                np.linalg.norm(h[-1] @ self.unembedding_matrix.T - h[0] @ self.unembedding_matrix.T)
            )
        return metrics

    def inference_metrics(self, h: np.ndarray, context: MetricContext) -> Dict[str, float]:
        """Family 10: prediction-readout metrics requiring token IDs."""

        metrics = {"confidence_slope": np.nan, "convergence_monitor": np.nan}
        if self.unembedding_matrix is None or context.truth_id is None:
            return metrics

        top1 = self.unembedding_matrix[context.truth_id]
        top2 = self.unembedding_matrix[context.wrong_id] if context.wrong_id is not None else top1
        gap = (h @ top1) - (h @ top2)
        if len(gap) > 1:
            metrics["confidence_slope"] = float(linregress(np.arange(len(gap)), gap).slope)

        ema = h[0].copy()
        monitors = []
        for idx in range(1, len(h)):
            ema = 0.1 * h[idx] + 0.9 * ema
            denom = (np.linalg.norm(h[idx]) + 1e-12) * (np.linalg.norm(ema) + 1e-12)
            monitors.append(np.dot(h[idx], ema) / denom)
        if monitors:
            metrics["convergence_monitor"] = float(np.mean(monitors))
        return metrics

    def compute_layer_metrics(
        self,
        h: np.ndarray,
        layer_index: int,
        context: Optional[MetricContext] = None,
    ) -> Dict[str, float]:
        """Compute all per-layer metrics for a single [T, D] trajectory."""

        context = context or MetricContext()
        metrics: Dict[str, float] = {}
        metrics.update(self.kinematic_metrics(h))
        metrics.update(self.volumetric_metrics(h))
        metrics.update(self.convergence_metrics(h))
        metrics.update(self.diffusion_metrics(h))
        metrics.update(self.recurrence_metrics(h))
        metrics.update(self.landmark_metrics(h, context))
        metrics.update(
            self.attractor_metrics(
                h,
                context.centroids_by_layer.get(layer_index)
                if context.centroids_by_layer
                else None,
            )
        )
        metrics.update(self.embedding_stability_metrics(h, context))
        metrics.update(self.information_metrics(h))
        metrics.update(self.inference_metrics(h, context))
        metrics["layer"] = float(layer_index)
        return metrics


def cross_layer_metrics(hidden_stack: np.ndarray) -> Dict[str, float]:
    """Compute metrics that depend on multiple layers."""

    metrics = {"interlayer_alignment": np.nan, "depth_acceleration": np.nan}
    if hidden_stack.ndim != 3 or hidden_stack.shape[0] < 2 or hidden_stack.shape[1] < 2:
        return metrics

    alignments = []
    speeds = []
    for layer_idx in range(hidden_stack.shape[0]):
        trajectory = hidden_stack[layer_idx]
        deltas = trajectory[1:] - trajectory[:-1]
        speeds.append(np.mean(np.linalg.norm(deltas, axis=1)))
        if layer_idx < hidden_stack.shape[0] - 1:
            next_trajectory = hidden_stack[layer_idx + 1]
            d1 = trajectory[1:] - trajectory[:-1]
            d2 = next_trajectory[1:] - next_trajectory[:-1]
            dot = np.sum(d1 * d2, axis=1)
            denom = np.linalg.norm(d1, axis=1) * np.linalg.norm(d2, axis=1)
            alignments.append(np.mean(dot / (denom + 1e-12)))

    if alignments:
        metrics["interlayer_alignment"] = float(np.mean(alignments))
    if len(speeds) > 1:
        metrics["depth_acceleration"] = float(linregress(np.arange(len(speeds)), speeds).slope)
    return metrics


def compute_all_metrics(
    hidden_stack: np.ndarray,
    metric_engine: Optional[TrajectoryMetrics] = None,
    context: Optional[MetricContext] = None,
) -> list[Dict[str, float]]:
    """Compute the full per-layer metric table for a [L, T, D] tensor."""

    metric_engine = metric_engine or TrajectoryMetrics()
    context = context or MetricContext()
    layer_shared = cross_layer_metrics(hidden_stack)
    rows = []
    for layer_idx in range(hidden_stack.shape[0]):
        row = metric_engine.compute_layer_metrics(hidden_stack[layer_idx], layer_idx, context=context)
        row.update(layer_shared)
        rows.append(row)
    return rows
