# Manuscript Corrections

## Core numerical corrections

- Abstract / summary paragraph (`trajectory_geometry_paper_2026-03-28.tex`, line 47):
  replace the `80--85%` variance claim with the audited wording from `variance_decomposition_summary.md`.
- Methods difficulty-bin description (`trajectory_geometry_paper_2026-03-28.tex`, line 154):
  the local EXP-15 archive uses quartile-based answer-magnitude bins labelled Small / Medium / Large / Extra Large.
  The current workspace contains zero literal `|answer| >= 10,000` examples.
- Finding 1 caption (`trajectory_geometry_paper_2026-03-28.tex`, line 262):
  update Deff means to G4 = 18.95, G1 = 3.39, d = 4.67.
- Finding 3 table and caption (`trajectory_geometry_paper_2026-03-28.tex`, lines 320-336):
  replace the `4.8 / 7.3 / 9.3 / 18.1` monotonic claim with the local quartile-bin audit:
  Small: d=7.75 (L0), Medium: d=7.18 (L2), Large: d=5.08 (L24), Extra Large: d=4.15 (L1).
- Finding 5 commitment paragraph (`trajectory_geometry_paper_2026-03-28.tex`, line 354):
  replace `Direct ~= 5` and `CoT ~= 11--14` with Direct mean/median 14.32 / 17 and CoT mean/median 40.01 / 39,
  and describe `time_to_commit` as a trajectory-index measure from a `window=6`, `stride=2` Rg-drop analysis.
- Finding 5 basis invariance paragraph and caption (`trajectory_geometry_paper_2026-03-28.tex`, lines 358-366):
  replace `r = 1.0` and `N ~= 1,350` with PCA r = 0.918 and SAE r = 0.576
  on the 100 nontrivial CoT trajectories.
- Finding 6 predictive table (`trajectory_geometry_paper_2026-03-28.tex`, lines 382-386):
  local layer-13 recomputation gives Direct length / geometry / combined AUCs
  0.822 / 0.955 / 0.973
  and CoT geometry AUC 0.823.
- Finding 7 variance paragraph (`trajectory_geometry_paper_2026-03-28.tex`, lines 400-506 and 583):
  use mean regime eta^2 27.78% at peak layers and regime share of explained variance 74.58%
  for the six requested regime-sensitive metrics, instead of `80--85%`.

## Figure integration

- Replace the placeholder boxes for Figures 1, 2, 3, 4, and 6 with `\includegraphics` calls to the new `figures_v2` PDFs.
- The paper still has no embedded figure environments for the regenerated Figure 5, Figure 7, or Figure 8.
  Those assets now exist on disk and can be cited or inserted in a follow-up pass if desired.
