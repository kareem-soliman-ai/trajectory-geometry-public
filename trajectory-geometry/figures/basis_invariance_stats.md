# Basis Invariance Statistics

## Status
Raw basis-comparison data found locally in `data/rq1_basis_invariance.csv`.
Direct rows are present but trivial (`t_c = 0` across all three bases), so the informative
scatterplots and correlations focus on the 100 CoT trajectories.

## Overall CoT correlations
- Raw vs PCA-16: Pearson r = 0.918, 95% CI [0.880, 0.944], Spearman rho = 0.885, N = 100
- Raw vs SAE-16k: Pearson r = 0.576, 95% CI [0.428, 0.694], Spearman rho = 0.422, N = 100

## Subgroup breakdown
- G3 (CoT fail): PCA r = 0.882, SAE r = 0.584, N = 50
- G4 (CoT success): PCA r = 0.976, SAE r = 0.548, N = 50

## Manuscript wording
The local archive supports:

> Commitment timing is robust to linear dimensionality reduction (raw vs PCA-16, r = 0.918)
> but only moderately aligned with the sparse nonlinear SAE basis (raw vs SAE-16k, r = 0.576).
