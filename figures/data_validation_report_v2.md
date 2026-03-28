# Data Validation Report V2

## Scope
This report audits the March 2026 manuscript against the local archived artefacts used by the
manuscript folder, with figures regenerated from the current workspace.

## Claim audit

| # | Claim | Status | Actual local value | Source |
| --- | --- | --- | --- | --- |
| 1 | Deff: G4 ~= 13.1 vs G1 ~= 3.4 at layer 14; d > 4.5 | DISCREPANCY | G4 = 18.95, G1 = 3.39, d = 4.67 | EXP-15 unified_metrics.csv |
| 2 | Rg: G4 ~= 40.2 vs G1 ~= 2.1 at layer 14; d > 4.0 | DISCREPANCY | G4 = 12.94, G1 = 9.20, d = 4.03 | EXP-15 unified_metrics.csv |
| 3 | Difficulty scaling: d = 4.8 / 7.3 / 9.3 / 18.1 | DISCREPANCY | Quartile-bin local audit gives Small: d=7.75 (L0), Medium: d=7.18 (L2), Large: d=5.08 (L24), Extra Large: d=4.15 (L1); literal threshold bins have counts Small=24, Medium=181, Large=95, Extra Large=0 | EXP-15 unified_metrics.csv |
| 4 | Direct prediction AUC = 0.898 +/- 0.03 | DISCREPANCY | 0.955 +/- 0.029 (layer 13, 10-feature local audit, N=282) | EXP-15 unified_metrics.csv |
| 5 | CoT prediction AUC = 0.772 | DISCREPANCY | 0.823 +/- 0.044 (layer 13, 10-feature local audit, N=300) | EXP-15 unified_metrics.csv |
| 6 | Length-only AUC = 0.645 +/- 0.04 | DISCREPANCY | Direct-only local audit gives 0.822 +/- 0.069 on the same complete-case layer-13 sample | EXP-15 unified_metrics.csv |
| 7 | Variance: regime proportion (specify conditions) | CONFIRMED BY RECOMPUTATION | Mean regime eta^2 all layers = 26.88%; peak-layer mean eta^2 = 27.78%; regime share of explained variance for six requested metrics at layers 10-16 = 74.58% | figures_v2/variance_decomposition_audit.csv |
| 8 | PCR gain Layer 0 = +0.119 | CONFIRMED | gain = 0.119 (raw 0.659 -> PCR 0.778) | EXP-19 analysis_19b/pcr_auc_comparison_qwen05b.csv |
| 9 | Commitment sharpness aggregate d = 9.83 | CONFIRMED | 9.831 | EXP-19 invariant_signatures.csv |
| 10 | 19 invariant signatures | CONFIRMED | 19 signatures | EXP-19 invariant_signatures.csv |

## Additional audited corrections
- Commitment timing in the local archive is not `Direct ~ 5` and `CoT ~ 11-14`.
  The current `time_to_commit` metric yields Direct mean/median 14.32 / 17 and CoT mean/median 40.01 / 39.
- Basis invariance is not perfect. The local basis table gives raw-vs-PCA Pearson r = 0.918
  and raw-vs-SAE Pearson r = 0.576 on the 100 nontrivial CoT trajectories.
- The Qwen PCR `+0.119` result and the Gemma PCR gains belong to different experiments and should not be conflated.
