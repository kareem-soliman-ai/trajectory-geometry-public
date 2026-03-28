"""Public analysis package for the Trajectory Geometry repository."""

from .contamination_filter import (
    find_clean_token_boundary,
    parse_numeric_answer,
    truncate_cot_response,
    truncate_direct_response,
    validate_response,
)
from .hidden_state_extraction import ExtractionConfig, build_prompt, extract_generation_trajectory
from .pcr import (
    CloudRegressor,
    estimate_sigma_from_cross_layer_variance,
    estimate_sigma_strategy_d,
    pcr_denoise_dataframe,
    shrinkage_denoise_dataframe,
)
from .statistical_analysis import (
    cohens_d,
    pcr_corrected_auc,
    permutation_test_mean_difference,
    stratified_logistic_auc,
    two_way_anova,
)
from .trajectory_metrics import TrajectoryMetrics, compute_all_metrics, cross_layer_metrics

__all__ = [
    "CloudRegressor",
    "ExtractionConfig",
    "TrajectoryMetrics",
    "build_prompt",
    "cohens_d",
    "compute_all_metrics",
    "cross_layer_metrics",
    "estimate_sigma_from_cross_layer_variance",
    "estimate_sigma_strategy_d",
    "extract_generation_trajectory",
    "find_clean_token_boundary",
    "parse_numeric_answer",
    "pcr_corrected_auc",
    "pcr_denoise_dataframe",
    "permutation_test_mean_difference",
    "shrinkage_denoise_dataframe",
    "stratified_logistic_auc",
    "truncate_cot_response",
    "truncate_direct_response",
    "two_way_anova",
    "validate_response",
]
