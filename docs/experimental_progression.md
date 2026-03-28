# Trajectory Geometry: Research History

**Date:** 2026-03-26 *(updated from 2026-03-16)*
**Experiment Range:** EXP-01 through EXP-23 (29 experiment entries)
**Author:** Kareem Soliman
**Companion Documents:** [Project Overview](./project-overview.md) | [Research Narrative](./research-narrative.md) | [Findings Catalogue](../reference/findings-catalogue.md)

---

## Overview

This document is the auditable, chronological record of the Trajectory Geometry research program — a project that began with the question "does cognitive operation leave a detectable shape in a language model's representation?" and evolved, through repeated failure and methodological reinvention, into a series of findings about how transformers physically navigate representational space when they reason.

Over 26 experiments spanning approximately 17 months, the research progressed from API output embeddings and clustering (EXP-01) through Probability Cloud Regression and multi-architecture invariant signature identification (EXP-19B) to the first domain-transfer test — empathy geometry (EXP-20). Each experiment is recorded with its research question, motivation, method, results, interpretation, and an explicit bridge to the next experiment. Phase-level narrative sections explain the larger pivots. The document captures both what was found and why the search evolved the way it did — including the seven failed or invalid experiments and one refuted domain-transfer attempt that shaped the eventual approach.

The theoretical origin was **Dynamic Semantic Geometry (DSG)** — a framework from psychology proposing that cognitive operators enact measurable geometric transformations on representational manifolds. The DSG framework was largely wrong in its specific predictions but generated the right questions. This history records how those questions were answered.

---

## Research Phase Map

| Exp | Phase | Date | Verdict | One-Line Summary |
|-----|-------|------|---------|-----------------|
| 01 | I — Shapes | Nov 2025 | Invalid | API output embeddings fail; topic noise dominates operator signal |
| 02 | I — Shapes | Nov 2025 | Invalid | Scalar warp magnitude loses direction; all operators look identical |
| 03 | II — Invariants | Nov 2025 | Failed | Regime-mapping finds topic confound, not operator signal |
| 04 | II — Detour | Nov 2025 | Success | Forcing multi-pass thinking (OG-MPT) improves accuracy 53%→77% |
| 05 | II — Detour | Nov 2025 | Invalid | Safety OG-MPT collapses at 0.5B scale; capacity floor discovered |
| 06 | II — Wilderness | Nov 2025 | Weak | Stricter validation (LOPO/LOCO) still yields F1~0.30-0.48 |
| 07 | II — Ceiling | Nov 2025 | Valid/Insufficient | Static stability 0.924 at L16, but separability F1~0.30 only |
| 08 | III — Dynamics | Nov 2025 | Breakthrough | Differential geometry metrics identify 9 computational regimes |
| 09 | III — Dynamics | Nov 2025 | Breakthrough | Speed/Curvature separates G4 vs G1 at d>3.0; reasoning≠hallucination |
| 09B | III — Dynamics | Nov 2025 | Failure | TinyLlama <1% accuracy; capability floor required for geometry |
| 10 | III — Dynamics | Nov 2025 | Failure | Model self-reports r≈0.00 with geometry; introspection unreliable |
| 11 | IV — Failure | Nov 2025 | Success | Dimensional Collapse: G4 D_eff≈13.1 vs G1≈3.4, d>4.5 |
| 12 | IV — Failure | Nov 2025 | Success | Fractal density and two-phase Explore→Commit transition confirmed |
| 13 | IV — Failure | Dec 2025 | Success | Two failure subtypes; AUC 0.898/0.772; "Retrieve-vs-Compute" |
| 14 | V — Paradigm | Dec 2025 | Breakthrough | 10/14 metrics flip sign; no universal success; regime-relative geometry |
| 15 | V — Validation | Dec 2025 | Success | Difficulty scaling d≈5→17; geometry AUC 0.79 > length AUC 0.77 |
| 16 | V — Scale | Dec 2025 | Success | Scale stable 0.5B→1.5B; Goldilocks zone; hallucination pipeline fixed |
| 16B | V — Scale | Dec 2025 | Success | Pythia-70m 0%→100% accuracy post parser fix; architecture independence |
| 17 | VI — Robustness | Feb 2026 | Partial | 3B scale tested; hardware ceiling limits full replication |
| 18 | VI — Robustness | Feb 2026 | Success | 54 metrics across 12 families formalized; TrajectoryMetrics class |
| 18B | VI — Robustness | Feb 2026 | Partial/Invalid | Pythia-70m data corrupted; pipeline hard constraints established |
| 19 | VI — Robustness | Feb 2026 | Breakthrough | 19 invariant signatures; Success Attractor confirmed; 3 architectures |
| 19B | VI — Robustness | Mar 2026 | Breakthrough | PCR corrects attenuation bias; AUC +0.119 at L0; G3 Position Index |
| 20 | VII — Domain Transfer | Mar 2026 | Refuted | Empathy geometry detected but attributable to prose domain, not perspective-taking |
| 21 | VIII — Cross-Arch | Mar 2026 | Success | Gemma 3 1B replicates arithmetic geometry; GemmaScope SAE features correlate but mechanistic claims constrained |
| 22 | VIII — Causal | Mar 2026 | Success | SAE feature ablation demonstrates construct-specific causal disruption; 15-30pp carry drop at s=0.25 with zero no-carry impact; Goodhart-resistant validator foundation |
| 23 | VIII — Causal | Mar 2026 | Success | Phase 3 completion: "Ghost Scalpel" feature 869 causes -36.3% accuracy collapse; feature 6340 confirmed inert on complex tasks |

---

## Phase I: The Intuition of Shapes (EXP-01–02)

The project grew from a theoretical framework in cognitive psychology. While developing PsychScope, a construct-based AI evaluation tool, the question arose: if language models instantiate cognitive operations — summarization, critique, elaboration — do those operations leave a measurable trace in the model's representational geometry? The framework proposed that different operators would produce characteristic "warp vectors" detectable in embedding space.

Two experiments tested this intuition using the tools most immediately available: API embeddings and open-weights hidden states. Both failed, but the failures were informative about what to look for instead.

---

### EXP-01: Geometric Signatures (Turn-Level)

**Date:** November 2025 | **Verdict:** Invalid

**Research Question:** Do different cognitive operators (Summarize, Critique, Elaborate) leave distinct geometric signatures in the embedding space of model outputs?

**Prior State:** The Dynamic Semantic Geometry framework predicted that operators produce characteristic "warp vectors" — the displacement between input and output embeddings. No prior measurement attempt had been made.

**Method:**
- **Model:** Gemini API (`text-embedding-004`)
- **Dataset:** 500 single-turn prompts (10 operators × 10 paraphrases × 5 topics)
- **Procedure:** Computed warp vectors $w_t = E(\text{response}_t) - E(\text{turn}_t)$; applied K-Means and HDBSCAN clustering; attempted GRU classification
- **Metrics:** AMI, Silhouette Score, Within/Between variance ratio

**Results:**
- K-Means AMI = **0.13** (barely above random); HDBSCAN found **0 clusters**
- Within-operator variance (0.296) **exceeded** between-operator variance (0.112)
- Two paraphrases of "Summarize" were more different from each other than "Summarize" was from "Critique"
- GRU predictor: 57–66% accuracy (vs 10% random), likely learned transition artifacts

**Interpretation — Invalid:** Output embeddings capture lexical and topic content, not the "cognitive act" of the operator. The warp vector conflates semantic content change with the operator transformation. No signal above noise.

**Bridge to EXP-02:** The failure of output embeddings motivated a move inside the model. If the signature exists at all, it must live in the *internal* hidden states before the collapse to text. EXP-02 accessed those states directly via an open-weights model.

---

### EXP-02: Latent Factors (Token-Level)

**Date:** November 2025 | **Verdict:** Invalid

**Research Question:** Do hidden state trajectories in open-weights model internals show operator-distinct warp traces, and are composite operators decomposable into single-operator components?

**Prior State:** EXP-01 showed API output embeddings were uninformative. The hypothesis was that the signal existed in hidden states before the final token projection.

**Method:**
- **Model:** Qwen2.5-0.5B (first use of open-weights model)
- **Metric:** Warp $W_t = ||h_t - h_{t-1}||_2$ (magnitude of consecutive hidden state change)
- **Procedure:** Ran single and composite operator prompts; extracted hidden states across all 24 layers; applied NMF to decompose warp traces; attempted reconstruction of composite operators
- **Analysis:** Layers 5–12 for "stable geometry"; Layer 24 as output

**Results:**
- **Universal signature:** All operators produced the same warp trace — a massive spike at $t=0$ followed by a flat line
- No distinct "oscillating" or "ramping" profiles for specific operators
- Linear reconstruction worked, but only because "spike+flat" reconstructs trivially from "spike+flat" basis vectors
- Middle layers (5–12): coherent geometry; final layer (24): chaotic vocabulary projection noise

**Interpretation — Invalid:** The **magnitude** metric discards all directional information. The $t=0$ spike is a universal transformer property (first embedding ingestion), drowning any operator-specific signal. The problem wasn't the model — it was the metric.

**Bridge to EXP-03:** The realization that direction matters more than magnitude motivated a search for regime-specific *directional* patterns. The model's "Thinking" tokens might move in different *directions* than "Speaking" tokens, even if the magnitudes looked the same. EXP-03 tested this by separating Listen, Think, and Speak phases.

---

## Phase II: The Invariant Trap and The Wilderness (EXP-03–07)

The next five experiments attempted to rescue the static operator hypothesis through increasingly rigorous measurement: regime separation, cross-validation schemes, adversarial datasets, and finally a large-scale definitive test. Two of the five were productive in unexpected ways (EXP-04 proved that forcing dynamics works; EXP-07 established the exact ceiling of static analysis). Three confirmed the dead end.

The critical intellectual shift happened slowly. The intuition was that the static vector exists but is hard to find. The data showed, repeatedly, that even when the technique worked technically, the practical signal was too weak to be useful. By EXP-07, the evidence was conclusive: *position* cannot explain *competence*.

---

### EXP-03: Regime Invariants

**Date:** November 2025 | **Verdict:** Failed

**Research Question:** Does a "Summarization vector" (or any operator vector) persist as the model transitions between Listen, Think, and Speak processing phases?

**Prior State:** EXP-02 showed magnitude metrics failed. The new hypothesis was that operators leave directional signatures that are invariant across processing regimes — a stable "abstract representation" of the operator.

**Method:**
- **Model:** Qwen2.5-0.5B
- **Dataset:** 10 operators × 2 topics (AI, Climate) × 10 paraphrases; balanced 50/50 topic split per operator
- **Procedure:** Extracted hidden states per regime (Listen/Think/Speak); trained linear probes (Logistic Regression) within- and cross-regime; CCA for shared subspace detection

**Results:**
- **Probes achieved 100% accuracy** — but this was **topic confounding**: the balanced design made any topic detector a perfect operator detector
- Cross-regime coupling maps: $R^2 < 0.07$ (< 7% variance explained)
- CCA alignment accuracy: **27.5%** (marginally above random 10%)
- No evidence of a shared geometric subspace between Listen and Think regimes

**Interpretation — Failed:** "Thought" is not a rotated version of instruction. The transition from processing input to generating output involves a fundamentally non-linear transformation, not a linear rotation or scaling. The "Static Vector" theory of operators was effectively dead.

**Bridge to EXP-04/08:** This killed the static position approach. But before fully pivoting, two experiments tested whether the *dynamic* failure was just a measurement problem (EXP-04 explored whether forced dynamics worked; EXP-06/07 made last-ditch static attempts). EXP-04 was a productive detour: it showed that *controlling* the dynamic process improves outcomes, even if we can't *measure* the static position.

---

### EXP-04: Operator-Gated Multi-Pass Thinking (OG-MPT)

**Date:** November 2025 | **Verdict:** Success

**Research Question:** If we cannot *find* the operator in the vector space, can we *force* it? Does explicitly structuring the model's internal monologue (Plan→Calculate→Verify) produce measurable performance gains?

**Prior State:** Experiments 1–3 failed to find static signatures. The complementary hypothesis: if dynamics matter (as Exp 3 implied), then *controlling* the dynamic process should improve behavior.

**Method:**
- **Model:** Qwen2.5-0.5B (Baseline vs. Orchestrator)
- **Dataset:** 60 rigorous prompts (Math, Constraints, Safety), generated via ChatGPT metaprompting and adversarially filtered
- **Procedure:** Heuristic task detector; Orchestrator forced multi-turn ChatML conversation per task type (Reasoning: Plan→Calculate→Verify→Answer; Constraints: List→Draft→Check→Answer; Safety: Identify→Check→Refuse/Answer)
- **Evaluation:** Strict regex + programmatic checking

**Results:**
- Overall accuracy: Baseline **53.3%** → Orchestrator **76.6%** (+23.3%)
- Constraint task: **40%** → **85%** (+45%)
- Reasoning (Math/Logic): +25%

**Interpretation — Success:** Cognitive capability is not just raw weights; it is **control flow**. A small model can substantially outperform its single-pass baseline when the *structure* of its thinking is made explicit. This proved that dynamics matter for performance even if the static vector theory is wrong.

**Bridge to EXP-05/08:** EXP-04 confirmed that forcing a thought structure works. EXP-05 tested if this generalized to safety. When EXP-05 failed, the combined signal was clear: the dynamic approach works for structured problems but requires model capacity. Eventually, EXP-08 returned to *measuring* these dynamics rather than forcing them.

---

### EXP-05: Safety Resilience (OG-MPT Expansion)

**Date:** November 2025 | **Verdict:** Invalid

**Research Question:** Does multi-pass gating (Plan→Check→Refuse/Answer) make a small model more resistant to jailbreaks?

**Prior State:** EXP-04 showed OG-MPT works for math and constraints. The question was whether it generalizes to safety — a more complex judgment task requiring the model to assess intent, not just execute a plan.

**Method:**
- **Model:** Qwen2.5-0.5B (Baseline vs. Orchestrator)
- **Dataset:** 54 prompts (benign controls + adversarial attacks: "Fictional Sandbox", "Authority Override")
- **Procedure:** Safety OG-MPT: Identify Harm → Check Policy → Refuse/Answer
- **Baseline ASR:** 56.8% (high vulnerability)

**Results:**
- OG-MPT **collapsed** into infinite repetition loops (e.g., repeating "Gründe" 75 times) or echoed the system prompt
- Only 15 valid OG-MPT samples vs 54 baseline samples — statistically meaningless comparison
- A code error also misclassified benign controls, rendering the helpfulness metric invalid

**Interpretation — Invalid:** The 0.5B model lacked the instruction-following bandwidth to maintain state across 3 internal passes. Complex dynamic architectures require a minimum capability threshold (likely 7B+ for stable multi-pass safety reasoning).

**Bridge to EXP-06:** EXP-05's failure, combined with EXP-04's success, clarified boundaries. The gating approach works when the task structure is well-defined and the steps are computationally within the model's reach. The focus returned to *measurement* rather than intervention — specifically, to finding what static signal remains once better controls are applied.

---

### EXP-06: Pilot Metric Validation (The Wilderness)

**Date:** November 2025 | **Verdict:** Inconclusive / Weak

**Research Question:** If topic confounding is removed via Leave-One-Paraphrase-Out (LOPO) and Leave-One-Content-Out (LOCO) cross-validation, does any meaningful static operator signal remain?

**Prior State:** EXP-03 found strong-looking results that were entirely confounded by topic. EXP-06 tested whether the signal survived strict holdout schemes.

**Method:**
- **Model:** Qwen2.5-0.5B
- **Design:** Pilot subset; 3 operator classes
- **LOPO:** Train on $N-1$ paraphrases, test on held-out paraphrase
- **LOCO:** Train on $N-1$ content types, test on held-out content
- **Control:** 10,000-shuffle permutation test for empirical chance baseline (F1 ~0.24)

**Results:**
- LOPO F1: **~0.30–0.48** (chance ~0.33)
- LOCO F1: **~0.45–0.50** (chance ~0.33)
- No single layer or metric strongly separated operators across all contexts
- Permutation baseline: F1 ~0.24 (confirming slight but real above-chance performance)

**Interpretation — Weak:** Signal exists marginally above random — but "distinct cognitive operators" should produce F1 > 0.8 to be practically useful. Extensive engineering of validation controls revealed a signal too weak to be the foundation of a measurement system.

**Bridge to EXP-07:** One definitive large-scale test was needed before abandoning the static approach entirely. If the signal at F1 ~0.30 was just due to small-N pilot noise, a dataset of N=2,000 should reveal the true ceiling. EXP-07 ran that test.

---

### EXP-07: Static Operator Geometry (The Ceiling)

**Date:** November 2025 | **Verdict:** Valid but Insufficient

**Research Question:** With N=2,000 samples, do operator centroids form stable, separable geometric structures in the model's hidden space?

**Prior State:** EXP-06 found weak signal. The hypothesis was that scale would reveal it. If the centroids are stable but just noisy with small N, a large dataset should close the gap.

**Method:**
- **Model:** Qwen2.5-0.5B
- **Dataset:** N=2,000 (10 operators × 20 content domains × 10 paraphrases)
- **Procedure:** "Speed Patch" directional delta extraction; centroid stability (mean pairwise cosine similarity across content domains); Nearest Centroid Classifier (NCC)

**Results:**
- **Layer 16 stability: 0.924** — operators do form stable abstract internal states
- NCC F1: **~0.30** (chance 0.10) — high stability but low separability
- Related operators (Critique vs Question) shared closer subspaces than unrelated pairs — semantic topology is present but fuzzy

**Interpretation — Valid but Insufficient:** Both hypotheses partially confirmed: operators *do* produce stable representations (H2, stability 0.924), but they are too overlapping to distinguish reliably (F1 0.30). This hit the exact ceiling of what static analysis can deliver. The "Summarize" direction exists — it's just too fuzzy for practical control.

**Bridge to EXP-08:** The decisive paradigm shift. EXP-07 proved position-based analysis is insufficient — not because the signal is absent, but because it is too blurry. The question became: if we measured *how* the state moves (speed, curvature, trajectory shape) rather than *where* it sits, would the signal be sharper? EXP-08 answered: yes, dramatically so.

---

## Phase III: The Pivot to Dynamics (EXP-08–10)

The shift from static to dynamic analysis was not a single decision but the accumulated weight of seven experiments that all pointed to the same ceiling. EXP-07 provided the final evidence. If static coordinates couldn't separate operators with d > ~0.3 F1, the measurement instrument was wrong — not the underlying reality.

The correct instrument turned out to be differential geometry: not *where* states are, but *how* they move. Speed, curvature, volume — physical properties of trajectory paths rather than positions in a fixed coordinate system.

EXP-08 established the measurement framework. EXP-09 applied it to success vs failure and found the most striking result in the entire program. EXP-09B and EXP-10 probed the boundaries of that finding.

---

### EXP-08: Trajectory Geometry

**Date:** November 2025 | **Verdict:** Foundational Success

**Research Question:** Can differential geometry metrics (Speed, Curvature, Radius of Gyration) applied to transformer hidden state sequences identify distinct computational "regimes"?

**Prior State:** EXP-07 established the ceiling of static analysis. The key insight from EXP-04 and EXP-07 combined: *forcing* dynamic structure improves performance, and static *positions* cannot distinguish operators. The natural next step: measure the *motion*.

**Method:**
- **Model:** Qwen2.5-0.5B
- **Metrics defined:**
  - **Speed:** $||h_t - h_{t-1}||_2$ — Euclidean step size
  - **Curvature:** Cosine angle between entrance and exit vectors at each state
  - **Radius of Gyration ($R_g$):** Volume of the trajectory cloud
- **Procedure:** K-Means on *dynamic metrics* (not states) to find regimes; Layer 13 analysis
- **EXP-08' sub-experiment:** Cue-word injection test — measured metric change before/after "Wait", "Therefore", "Plan"

**Results:**
- K=9 clusters found with silhouette score **0.179** and **69% predictability** (Layer 13)
- Distinct modes of motion confirmed — the model switches between recognizable "computational modes"
- **EXP-08' result: FAILED** — cue-word injection did not reliably shift geometric regime ($p > 0.05$)

**Interpretation — Foundational Success:** The right measurement instrument was found. Dynamic trajectory metrics reveal computational structure that static position metrics completely missed. However, dynamics are emergent — they cannot be triggered by individual tokens.

**Bridge to EXP-09:** EXP-08 could identify *that* regimes exist but not which were "good" or "bad." EXP-09 applied these metrics to the most important question: does a *correct* thought look different from a *wrong* one?

---

### EXP-09: Geometry-Capability Correlation

**Date:** November 2025 | **Verdict:** Breakthrough

**Research Question:** Do the dynamic trajectory metrics from EXP-08 distinguish successful reasoning (G4: CoT Success) from failed direct retrieval (G1: Direct Failure)?

**Prior State:** EXP-08 had the metrics but no outcome labels. EXP-04 had outcome variation (Success vs Failure) but no trajectory metrics. EXP-09 combined the two.

**Method:**
- **Model:** Qwen2.5-0.5B
- **Dataset:** 300 multi-step arithmetic problems (e.g., $(A \times B) + C$)
- **Groups:**
  - **G1 (Direct Failure):** Answers immediately and incorrectly
  - **G4 (CoT Success):** Uses chain-of-thought and answers correctly
- **Metrics:** Speed, Directional Consistency, Stabilization Rate
- **Control:** Window size equalized across groups to prevent length confound

**Results:**
- **Speed:** G4 is **3–4× faster** than G1, Cohen's $d > 3.0$ (Layer 24) — successful thought moves with more energy
- **Directional Consistency:** G1 ~0.5 (near-straight path); G4 ~0.05 (winding path); $d = 2.6$
- **Stabilization:** G4 stabilizes in final layers (convergence); G1 destabilizes (wandering noise)
- Effect sizes exceeded all conventional thresholds ($p < 0.001$, permutation-tested)

**Interpretation — Breakthrough:** *Hallucination is geometrically a straight line. Reasoning is a winding path.* The model literally traverses more representational volume when it reasons correctly. A "System 1" direct failure fires ballistically to a wrong answer. A "System 2" CoT success diffuses through semantic space, exploring before converging. This was the first result that could be used to detect reasoning quality from latent geometry alone — without reading the output.

**Bridge to EXP-09B/10:** Two boundary conditions needed testing. First: does this replicate on a different architecture? Second: does the model *know* its own geometry — can it introspect on these states?

---

### EXP-09B: Cross-Model Replication (TinyLlama)

**Date:** November 2025 | **Verdict:** Failure (Technical)

**Research Question:** Do the geometric signatures of EXP-09 replicate on TinyLlama-1.1B-Chat?

**Prior State:** EXP-09 found strong signatures in Qwen-0.5B. Cross-architecture replication would support the claim that these are universal computational properties.

**Method:**
- **Model:** TinyLlama-1.1B-Chat
- **Dataset:** 300 multi-step arithmetic problems (same as EXP-09)
- **Same metrics:** Speed, Curvature, Stabilization

**Results:**
- G4 (CoT Success) = **<1%** across 300 problems (confirmed by manual audit)
- No statistics possible with 1–2 success samples in 300
- The model could not generate coherent chains of thought

**Interpretation — Failure (Technical):** Geometry requires a capability floor. If the model cannot reason, there is no reasoning geometry to measure. This is not a failure of the geometric framework — it is a boundary condition: the framework applies to models with sufficient capacity to actually execute the task.

**Bridge to EXP-10:** Returned to Qwen-0.5B. The next question: if external observers (us) can detect geometry, can the model detect it *about itself*? EXP-10 tested whether verbal self-reports correlated with measured geometric states.

---

### EXP-10: Self-Report Consistency

**Date:** November 2025 | **Verdict:** Failure of Hypothesis

**Research Question:** Do the model's verbal self-reports of effort, certainty, exploration, and smoothness correlate with its objective geometric trajectory metrics?

**Prior State:** EXP-09 showed geometry tracks reasoning quality. If the model's internal geometric state is meaningfully structured, it might have implicit introspective access. This would enable the model to self-monitor and self-correct.

**Method:**
- **Model:** Qwen2.5-0.5B
- **Protocol:**
  1. **Solve:** Generate chain-of-thought solution
  2. **Measure:** Compute objective Speed, Curvature, Stabilization
  3. **Introspect:** Prompt model to rate Effort, Certainty, Exploration, Smoothness (1–5 scales)
  4. **Perturb:** Re-ask ratings after minimal irrelevant input ("ACK") to test stability

**Results:**
- **Effort vs Speed:** $r \approx 0.00$
- **Certainty vs Stabilization:** $r \approx -0.01$
- **Exploration vs Directional Consistency:** Weak, noisy, non-significant
- Self-reports were **inconsistent**: perturbing with "ACK" often changed ratings significantly
- Reports were **performance-biased**: model rated itself high on Certainty regardless of internal state (RLHF training artifact)

**Interpretation — Failure of Hypothesis:** Introspection is a hallucination. The model's verbal self-monitoring is not a readout of its geometric state — it is another generation, driven by surface text patterns rather than latent dynamics. We cannot ask the model "Are you stuck?" We must *measure* whether it is stuck.

**Bridge to EXP-11:** With introspection ruled out as a monitoring channel, the focus shifted entirely to external geometric measurement. EXP-11 deepened the metric suite to characterize the *structure* of failure and success more precisely.

---

## Phase IV: The Richness of Failure (EXP-11–13)

The breakthrough of EXP-09 showed that geometry separates success from failure. The next three experiments asked a harder set of questions: *what kind of failure?* *What kind of success?* And *can we predict outcome purely from geometry?*

These experiments introduced the metric suite that defined the project's analytical vocabulary: Effective Dimension, Tortuosity, Fractal Dimension, and the failure subtype taxonomy. The key discovery was that failures are not monolithic — they have distinct geometric signatures that reveal different *mechanisms* of failure, not just different *degrees* of failure.

---

### EXP-11: Extended Geometric Suite

**Date:** November 2025 | **Verdict:** Success

**Research Question:** Can an extended metric suite (Effective Dimension, Tortuosity, Directional Autocorrelation) reveal additional structural properties of success vs failure trajectories?

**Prior State:** EXP-09 established Speed and Curvature as predictors. The hypothesis was that these were proxies for deeper topological properties — specifically, the dimensionality and path efficiency of the trajectory manifold.

**Method:**
- **Model:** Qwen2.5-0.5B
- **Dataset:** 300 arithmetic problems (EXP-09 dataset)
- **New Metrics:**
  - **Effective Dimension ($D_{eff}$):** PCA participation ratio — how many principal components are needed to explain the trajectory variance
  - **Tortuosity ($\tau$):** End-to-end distance / total arc length
  - **Turning Angle:** Mean angular change between consecutive step deltas
  - **Directional Autocorrelation:** Whether one step direction predicts the next

**Results:**
- **Dimensional Collapse (G4 vs G1):** G4 $D_{eff} \approx 13.1$; G1 $D_{eff} \approx 3.4$; Cohen's $d > 4.5$
- **Critical nuance:** Even *failed* CoT (G3) maintained $D_{eff} \approx 13.9$ — CoT prompting forces high-dimensional engagement regardless of outcome
- **Tortuosity:** G4 $\tau \approx 0.04$ (extremely winding); G1 $\tau \approx 0.40$ (relatively straight)
- Turning Angle and Directional Consistency were highly correlated ($r > 0.9$) — redundant information

**Interpretation — Success:** The landmark discovery: **Dimensional Collapse in Failure**. Successful reasoning is not just faster or more curved — it is fundamentally higher-dimensional. A failing direct answer collapses into ~3 effective dimensions (a thin "line" through representational space). Successful CoT reasoning uses ~13 dimensions. The theoretical reframe: CoT is not "extra compute" but **dimensional expansion** — unfolding compressed representations into a space where they can be manipulated.

**Bridge to EXP-12:** EXP-11 established dimensionality as a key predictor. EXP-12 probed the *texture* of that high-dimensional space: is it fractally complex? Does it have temporal structure (a "beginning" and "end" phase)?

---

### EXP-12: Advanced Geometric Diagnostics

**Date:** November 2025 | **Verdict:** Success

**Research Question:** Does successful reasoning show higher fractal complexity and a measurable two-phase (Explore→Commit) temporal structure?

**Prior State:** EXP-11 showed that success is high-dimensional. The hypothesis was that successful trajectories are not just high-dimensional but *fractally dense* — they revisit and re-evaluate representational regions, rather than simply traversing high-dimensional space in a straight line.

**Method:**
- **Model:** Qwen2.5-0.5B
- **Advanced Metrics:**
  - **Fractal Dimension ($D_f$):** Space-filling complexity of the trajectory
  - **Intrinsic Dimension (MLE):** Minimum variables to describe the manifold
  - **Convergence Slopes:** Rate at which tokens approach the final hidden state
  - **Early-Late Ratio:** Trajectory "energy" in first vs second half
  - **RQA (Recurrence Quantification):** Repeating patterns (determinism, laminarity)

**Results:**
- **Fractal Complexity:** G4 $D_f \approx 2.0$ vs G1 $D_f \approx 1.7$ — reasoning is fractally denser
- **Two-Phase Transition:** Success showed high-dimensional expansion (Exploration) followed by sharp contraction (Commitment); failure showed flat convergence or divergent wandering
- **Layer-wise Peak:** Geometric signal (G4 vs G1 delta) peaked in **Layers 10–16** — reasoning is "born" in the computational middle of the network
- **RQA finding:** Determinism/Laminarity metrics too noisy in 0.5B model for reliable use

**Interpretation — Success:** Reasoning is not just high-dimensional — it is *fractally dense*. The model iterates and re-evaluates in a way that fills the local representational volume. The two-phase profile (Explore→Commit) was confirmed as a structural feature of success, not an artifact. The middle layers (10–16) host the reasoning computation; early layers parse input, late layers output.

**Bridge to EXP-13:** The next question: are all failures the same? The failure cases had been treated as a single group. But the emerging picture — some failures are flat, some wander — suggested two mechanistically distinct failure modes. EXP-13 mined the data for these subtypes without collecting new data.

---

### EXP-13: Regime Mining and Failure Subtyping

**Date:** December 2025 | **Verdict:** Success

**Research Question:** Do CoT failures (G3) cluster into mechanistically distinct subtypes? And how accurately does trajectory geometry predict correctness?

**Prior State:** EXP-12 described what success looks like. The failure cases had been analyzed only as a single group. The emerging intuition was that not all failures were the same: some were "never engaged" and some were "actively lost."

**Method:**
- **Model:** Qwen2.5-0.5B
- **Strategy:** Deep mining of existing dataset (no new data collection)
- **Clustering:** K-Means ($k=3$) on 31-metric suite
- **Phase Transition Detection:** Sliding window "Dimension Drop" (early dim − late dim)
- **Predictive Modeling:** Logistic Regression on trajectory metrics (5-fold cross-validation)
- **Analysis Layer:** Layer 13

**Results:**
- **Two failure subtypes confirmed:**
  - **Subtype A — "The Broken Engine" (Collapsed Failure):** High tortuosity, low $D_{eff}$; model never engaged the reasoning regime; geometrically indistinguishable from a Direct answer
  - **Subtype B — "The Lost Wanderer" (Incoherent Failure):** High expansion, high dimensionality, but negative convergence; model "thought hard" but drifted away from the solution basin
- **Direct Success (G2) profile:** Higher speed, straighter paths, dramatically lower $D_{eff}$ than CoT Success — confirmed "Retrieve-and-Commit" signature
- **G4 Commitment:** Significant Dimension Drop in second half of trajectory — the geometric "Consensus" signature
- **Predictive Power:** AUC **0.898** for Direct answers, **0.772** for CoT; vs. prompt-type alone AUC ~0.63

**Interpretation — Success:** Geometry can now diagnose *mechanism* (Retrieval vs Reasoning) and *failure mode* (Collapse vs Confusion) without reading the output text. The commitment signature (Dimension Drop) was confirmed as a real structural feature of success. The failure taxonomy added diagnostic richness: a "Collapsed" failure needs regime engagement; a "Wandering" failure needs commitment.

**Bridge to EXP-14:** The 10-metric suite was proving reliable. The next question was whether a more comprehensive metric expansion would reveal a *universal* success signature — one that works regardless of prompting regime. EXP-14 tested this hypothesis directly, and its answer was the most important result in the project.

---

## Phase V: Paradigm Shift and Validation (EXP-14–16B)

The transition from Phase IV to Phase V was driven by a specific hypothesis: if geometry predicts success this well within one regime, perhaps there is a *universal* success signature that works across regimes. EXP-14's refutation of that hypothesis — finding instead that metrics flip sign — was the project's most important single result. It reframed every subsequent analysis.

EXP-15 stress-tested the difficulty dimension. EXP-16 and EXP-16B extended the findings to new scales and architectures, encountering and solving the hallucination contamination problem along the way.

---

### EXP-14: Comprehensive Metric Expansion

**Date:** December 2025 | **Verdict:** Breakthrough

**Research Question:** Does a 33-metric suite computed across all 28 layers reveal a universal geometric signature of success — one that holds regardless of prompting regime?

**Prior State:** EXP-13 established high predictive accuracy. The hypothesis (H1) was that success would look the same whether achieved via CoT or Direct answering. The project's first strong universal claim.

**Method:**
- **Model:** Qwen2.5-0.5B
- **Full Depth:** 28 layers
- **New Metrics (33 total):**
  - `cos_to_running_mean` (Coherence)
  - `time_to_commit` (Phase timing — token of maximum $R_g$ drop)
  - `msd_exponent` (Diffusion character)
  - `interlayer_alignment` (Cross-layer synchrony)
  - `spectral_entropy` (Path complexity)
- **Full trajectory capture** (up to 128 tokens) replacing prior 32-token window

**Results:**
- **Paradigm shift:** **10 of 14 key metrics showed opposite effects** for CoT vs Direct success
  - CoT Success: Lower Speed, Smaller $R_g$, Higher Coherence (Focusing)
  - Direct Success: Higher Speed, Larger $R_g$, Lower Coherence (Ballistic Retrieval)
- **`time_to_commit`:** Direct Success commits at ~5 tokens; CoT Success at ~11 tokens; Failures commit late or not at all
- **Layer depth profiles:**
  - **Layers 0–7:** Regime detection (CoT vs Direct), Cohen's $d = 6$–$8$
  - **Layers 10–14:** Within-regime success prediction, $d = 1.0$–$2.2$
  - **Layers 20–24:** Commitment timing

**Interpretation — Breakthrough:** **There is no universal success signature.** H1 was falsified. Success geometry is regime-dependent — what looks "good" under CoT criteria looks like failure under Direct criteria. A "correct" Direct answer looks like a failed CoT trajectory. This fundamentally changes how monitoring must be designed: any success detector must first identify the active computational regime before assessing trajectory quality. The regime-specific monitoring architecture became the project's central methodological principle.

**Bridge to EXP-15:** With the regime-relativity principle established, the next challenge was the length confound — the objection that geometry is just a proxy for token count. EXP-15 directly tested this.

---

### EXP-15: Stress-Testing the Phase Transition

**Date:** December 2025 | **Verdict:** Success

**Research Question:** Does geometric expansion scale with problem difficulty (independent of trajectory length), and does geometry outpredict length in success classification?

**Prior State:** EXP-14 confirmed regime-relative geometry. The most natural skeptical objection: "CoT works because longer outputs contain more information; geometry is just a proxy for length." EXP-15 tested this directly.

**Method:**
- **Model:** Qwen2.5-0.5B
- **Dataset:** 300 problems, stratified by difficulty (Small to Extra Large by operand size)
- **Ablation:** Geometric metrics vs response length in logistic regression (AUC comparison)
- **Anomaly Analysis:** "Direct-Only Successes" — cases where CoT failed but Direct answered correctly

**Results:**
- **Difficulty Scaling:** $R_g$ effect size (G4 vs G1) scaled monotonically with difficulty:
  - Small: $d \approx 5.0$ → Extra Large: $d > 17.0$ (Layer 4)
- **Geometry vs Length:**
  - Geometry only: AUC **0.79**
  - Length only: AUC **0.77**
  - Geometry subsumes the predictive value of length
- **"Overthinking" signature:** Direct-Only Successes showed artificial dimensionality expansion on problems the model had already memorized — CoT introduced noise by over-unfolding a memorized answer

**Interpretation — Success:** The model selectively allocates geometric expansion proportional to problem complexity — a **resource-rational** strategy. Geometry is not downstream of length; it is the *mechanism* that length reflects. Long trajectories predict success only insofar as they contain genuine dimensional complexity. The overthinking signature further confirmed this: expansion that isn't resolution-driven is noise.

**Bridge to EXP-16:** The next critical test: does this replicate on a larger model (1.5B parameters) and a different architecture (Pythia)? EXP-16 and 16B ran those replications.

---

### EXP-16 / EXP-16-Salvage / EXP-16B: Scale and Architecture Pivot

**Date:** December 2025 | **Verdict:** Success

**Research Questions:** (a) Do geometric signatures scale from 0.5B to 1.5B parameters? (b) Are these signatures architecture-independent (Qwen vs Pythia)?

**Prior State:** EXP-15 established the geometric framework on Qwen-0.5B. Architecture independence was a necessary condition for generalizability claims. Qwen-1.5B and Pythia-70m were the test cases.

**Method:**
- **Models:** Qwen2.5-1.5B (Exp 16), Pythia-70m (Exp 16-Salvage)
- **Dataset:** Multi-step arithmetic (same benchmark)
- **New pipeline requirement:** Boundary-aware hallucination truncation (both models exhibited "Runaway Hallucination" — generating new questions after answering)
- **Pythia salvage:** Re-ran with boundary-aware parser after discovering the original 0% accuracy was a parsing artifact

**Results:**
- **Qwen-1.5B (post-truncation):** Replicated all primary signatures — dimensional expansion, regime-relative geometry, scale-stable effect sizes
- **Pythia-70m salvage:** Initial **0% accuracy** → **100% accuracy** post-parser fix; geometric signatures confirmed for architecture independence at 70M parameters
- **Cross-architecture finding:** Success geometry is relative to the model's dominant failure mode:
  - Qwen-0.5B fails by *collapsing* (repetition/looping) → success looks like **expansion**
  - Qwen-1.5B / Pythia fail by *wandering* (hallucination/gibberish) → success looks like **compression**
  - **Goldilocks Zone:** Success is controlled expansion — between the extremes of collapse and wandering

**Interpretation — Success:** Scale stability confirmed from 70M to 1.5B. Architecture independence confirmed across LLaMA-style (Qwen) and GPT-style (Pythia) transformers. The Goldilocks Zone finding deepened the regime-relativity principle: "success" is not just regime-dependent but model-dependent, calibrated against the model's characteristic failure mode.

**Bridge to EXP-17:** With the core findings replicated across scales and architectures, the project shifted to validation robustness. EXP-17 extended the scale ladder to 3B parameters and tested the full 33-metric pipeline on the larger model.

---

## Phase VI: Robustness, Scaling, and Denoising (EXP-17–19B)

With the core geometric framework validated through EXP-16B, the final phase addressed four questions: (1) Does the scale ladder extend to 3B? (2) Can the metric suite be standardized for replication? (3) Are the quantitative findings robust across architectures and difficulty bins in a single large experiment? (4) Are the raw AUC estimates accurate, or are they biased by measurement noise?

EXP-17 encountered hardware limits at 3B. EXP-18 formalized the metric framework. EXP-18B stress-tested the pipeline and discovered data integrity issues. EXP-19 was the definitive multi-architecture robustness study. EXP-19B applied Probability Cloud Regression to correct attenuation bias — revealing that the signal was even stronger than previously measured.

---

### EXP-17: Baseline Replication & Multi-Mode Prompting

**Date:** February 2026 | **Verdict:** Partial

**Research Question:** Do the EXP-09 geometric signatures replicate when computed via the full 33-metric pipeline on Qwen2.5-3B? Does the scale ladder extend to 3B parameters?

**Prior State:** EXP-16B confirmed signatures up to 1.5B. The next natural scale step was 3B.

**Method:**
- **Model:** Qwen2.5-3B-Instruct
- **Phase 17A:** Direct replication of EXP-09 with 300 problems; full 33-metric pipeline
- **Phase 17B:** 8-mode multi-mode prompting — same content across different computational modes
- **Hardware:** AMD RX 5700 XT, 8GB VRAM via DirectML

**Results:**
- Regime-relative geometry confirmed on 3B scale
- 33-metric pipeline successfully computed on 3B model
- **Limitation:** Hardware ceiling at 8GB VRAM with DirectML began causing instability at 3B scale; full replication dataset was incomplete

**Interpretation — Partial:** The 3B model appears consistent with smaller Qwen models, but hardware constraints prevented statistically robust replication. The scale ladder extends to 3B in principle but is not conclusively validated.

**Bridge to EXP-18:** The hardware ceiling at 3B redirected focus to formalizing the metric framework for the validated 0.5B–1.5B range. EXP-18 consolidated the metric suite into a replication-ready standard.

---

### EXP-18: Consolidated Metric Suite

**Date:** February 2026 | **Verdict:** Success (Infrastructure)

**Research Question:** Can the full geometric metric suite be formalized into a standardized, replication-ready framework for cross-experiment and cross-model comparison?

**Prior State:** The metric suite had grown organically through EXP-01 to EXP-16B from 3 metrics to ~33, spread across multiple scripts with inconsistent implementations. This created barriers to replication and cross-experiment comparison.

**Method:**
- **Model:** Qwen2.5-0.5B (existing hidden states from EXP-09 and EXP-14 combined)
- **Scope:** 54 distinct metrics grouped into 12 conceptual families:
  - Kinematic, Volumetric, Convergence, Diffusion, Spectral, RQA
  - Cross-Layer, Landmark, Attractor, Embedding Stability, Information, Inference
- **Output:** `TrajectoryMetrics` class (`metric_suite.py`) for comprehensive geometric profiling

**Results:**
- 54 metrics formalized with consistent implementations across all families
- Combined dataset (EXP-09 + EXP-14 hidden states) successfully processed through the new pipeline
- Framework validated: metrics reproduce prior results with matching effect sizes

**Interpretation — Success (Infrastructure):** The project transitioned from exploratory data analysis to a standardized analytical framework. The 54-metric suite enables rigorous cross-experiment and cross-model comparison with consistent implementations. This was a necessary precondition for the large-scale EXP-19 replication study.

**Bridge to EXP-18B:** Before deploying the new suite on all three architectures, a multi-architecture stress test was needed. EXP-18B ran the suite on Pythia-70m, Qwen-0.5B, and Qwen-1.5B simultaneously — and discovered critical pipeline vulnerabilities.

---

### EXP-18B: Scaling Geometry (Pipeline Stress Test)

**Date:** February 2026 | **Verdict:** Partial / Invalid (Pythia)

**Research Question:** Does the 54-metric suite compute correctly across all three architectures simultaneously, and what attractor dynamics emerge at scale?

**Prior State:** EXP-18 formalized the metric suite. EXP-18B would apply it to all three validated architectures using existing hidden state data from EXP-14 (Qwen-0.5B), EXP-16 (Pythia-70m), and EXP-16B (Qwen-1.5B).

**Method:**
- **Models:** Pythia-70m, Qwen2.5-0.5B, Qwen2.5-1.5B
- **Reused hidden states** from prior experiments
- **Focus:** 57 metrics across 12 families; attractor dynamics, distance to success centroids

**Results:**
- **Pythia-70m data corruption discovered:** Hidden states loaded as 1,536-dim instead of expected 512-dim — completely invalidating all Pythia-70m results
- **Broadcasting crash:** `attractor_metrics` crashed when $T_{ref} = D$ (ambiguous numpy broadcasting)
- **Windows multiprocessing overhead:** Heavy `torch`/`transformers` imports at global scope in `spawn` mode caused memory pressure and UI freezing
- **No persistence:** Any crash wiped all progress — entire pipeline redesigned for atomic operations and incremental CSV appending

**Interpretation — Partial/Invalid:** EXP-18B functioned as an involuntary pipeline stress test. Pythia-70m results were invalidated by data corruption. But the experiment established hard constraints that made all subsequent work more robust: mandatory pre-flight tensor shape validation, explicit dimension handling for centroids, lazy imports for multiprocessing, process isolation to prevent memory leakage.

**Bridge to EXP-19:** EXP-18B's failures clarified exactly what the robust pipeline required. EXP-19 ran a clean, fresh data collection on three architectures — upgrading Pythia from 70m to 410m to match the 24-layer depth of Qwen-0.5B — using all the engineering lessons from EXP-18B.

---

### EXP-19: Robustness Replication

**Date:** February 2026 | **Verdict:** Breakthrough

**Research Question:** Do the core geometric signatures replicate across three disparate architectures (Qwen-0.5B, Qwen-1.5B, Pythia-410m) on a fresh 400-problem dataset with strict anti-contamination controls, and can a set of architecture-invariant signatures be identified?

**Prior State:** EXP-18B had identified pipeline vulnerabilities; EXP-18 had formalized the metric suite. EXP-19 was the definitive validation study — new data collection, upgraded architectures, strict guardrails.

**Method:**
- **Models:** Qwen2.5-0.5B, Qwen2.5-1.5B, Pythia-410m *(upgraded from 70m to match 24-layer depth)*
- **Dataset:** 400 problems, balanced across 4 difficulty bins (Small, Medium, Large, Negative)
- **Key Design Improvements:**
  - **Anti-contamination pipeline:** Multi-stage guardrails (prompt engineering, generation stop sequences, post-generation text truncation, boundary detection) eliminating "runaway hallucination"
  - **Few-shot calibration:** CoT-guided few-shot examples enabling non-zero accuracy for the 410m model
  - **Physical Trajectory Preservation:** 1,200 full 200-token trajectories stored on external HDD; semantic and post-answer geometry captured

**Results — Accuracy:**

| Model | UltraSmall | Small | Overall CoT |
| :--- | :--- | :--- | :--- |
| **Qwen2.5-1.5B** | 100.0% | 100.0% | **95.0%** |
| **Qwen2.5-0.5B** | 100.0% | 50.0% | **45.0%** |
| **Pythia-410m** | 25.0% | 0.0% | **5.0%** |

**Results — Top Invariant Predictors (G4 vs G1, cross-architecture):**

| Metric | Avg Cohen's $d$ | Peak |
| :--- | :--- | :--- |
| `phase_count` | 31.66 | Pythia-410m L23: $d = 70.11$ |
| `radius_of_gyration` | 13.99 | Qwen-1.5B L20: $d = 11.14$ |
| `effective_dimension` | 12.01 | — |
| `commitment_sharpness` | 9.83 | — |
| `tortuosity` | 6.93 | — |
| `direction_consistency` | 6.45 | — |

- **19 architecture-invariant signatures** identified (|Cohen's $d$| > 2.0 across all three models)
- **"The Snap" phenomenon:** Sharp phase transition (Commitment Sharpness) at the moment the model locks onto the correct solution — architecturally universal
- **Physical trajectory persistence:** Geometric signals survive beyond semantic answer boundaries; post-answer drift correlates with preceding reasoning quality

**Interpretation — Breakthrough:** The core geometric signatures are **architecturally invariant** — they hold across both LLaMA-style (Qwen) and GPT-style (Pythia) transformers, spanning 410M to 1.5B parameters. The Success Attractor — a tight, reproducible geometric manifold that successful reasoning trajectories converge onto — is a real, measurable, and universal feature of transformer computation.

**Bridge to EXP-19B:** The raw AUC estimates from EXP-19 (within-CoT AUC ~0.78) might themselves be biased. Geometric metrics computed from short token sequences have high per-token variance, which compresses estimated AUC toward chance — a statistical phenomenon known as attenuation bias. EXP-19B applied Probability Cloud Regression to correct for this.

---

### EXP-19B: Probability Cloud Regression (PCR) Reanalysis

**Date:** March 2026 | **Verdict:** Breakthrough

**Research Question:** Are the AUC estimates from EXP-19 accurate, or are they deflated by measurement noise? Does PCR denoising reveal a stronger underlying signal — and can regime-quality variance be formally decomposed?

**Prior State:** EXP-19 established within-CoT AUC ~0.78 at Layer 16. However, short token sequences produce noisy geometric feature estimates. If noise attenuates the true signal, raw AUC underestimates the model's true predictability.

**Method:**
- **Data:** EXP-19 hidden states (Qwen-0.5B, Qwen-1.5B)
- **PCR Methodology:**
  1. **Uncertainty estimation:** Per-trajectory $\sigma$ estimated from the standard deviation of each metric across layers of that trajectory
  2. **Denoising (Mode B):** Features projected onto a "true" manifold via `CloudRegressor`, anchored to sample ID (leakage-free — not anchored to correctness labels)
  3. **Re-prediction:** Logistic regression re-run on denoised features
- **Variance Decomposition:** Two-way ANOVA (Regime × Correctness) on 20+ metrics across all layers
- **Position Index:** Measure of where G3 (CoT Failure) sits on the axis from G1 (Direct Failure, PI=0) to G4 (CoT Success, PI=1)

**Results — PCR AUC Recovery (Qwen-0.5B):**

| Layer | Raw AUC | PCR-Corrected AUC | Gain |
| :--- | :--- | :--- | :--- |
| 0 | 0.659 | 0.778 | **+0.119** |
| 5 | 0.700 | 0.779 | **+0.079** |
| 16 | 0.799 | 0.779 | −0.020 *(slight over-smooth)* |

**Results — Regime-Quality Decomposition:**
- **Main Effect (Regime):** ~80–85% of total geometric variance — Direct and CoT trajectories are physically separated in embedding space
- **Main Effect (Quality):** η² ≈ 0.10 — robust within-regime signal persisting after regime is controlled
- **Within-CoT AUC:** Qwen-0.5B **0.78** (L16); Qwen-1.5B **0.74** (L26); far above regime-only baseline of 0.50
- **Interaction signatures — sign flips:**
  | Metric | Layer | Direct ($d$) | CoT ($d$) |
  |--------|-------|------------|---------|
  | `full_time_to_commit` | 3 | +1.50 | −0.38 |
  | `clean_cos_slope_to_final` | 4 | −0.33 | +0.46 |

**Results — G3 Position Index:**
- **Aggregate PI ≈ 0.033** — CoT Failures geometrically almost identical to Direct Failures overall
- **Layer-resolved:** G3 PI ~1.0 in early layers — CoT failures successfully *enter* the CoT attractor but fail to *commit* to the Success Centroid

**Interpretation — Breakthrough:**
1. **Proto-attractors form at Layer 0.** Raw measurements obscure this due to high per-token variance; PCR reveals the signal is present immediately. True predictability is likely **>0.85** if noise were perfectly eliminated.
2. **Deep layers (L16+) already have high SNR** — PCR provides minimal marginal gain there, confirming that commitment geometry solidifies progressively through the network.
3. **G3 failure is a commitment failure, not a regime-entry failure.** CoT failures enter the CoT manifold correctly (early-layer PI ~1.0) but cannot converge onto the Success Centroid. This refines the failure taxonomy established in EXP-13.
4. **The Success Attractor is real and distinct** from the CoT Regime Attractor. Within-regime geometry predicts success with AUC 0.78 — far above the 0.50 regime-only baseline.

**Bridge to EXP-20:** With the arithmetic reasoning arc closed and invariant signatures confirmed, the natural next question was domain transfer: does trajectory geometry generalise beyond arithmetic? EXP-20 targeted cognitive empathy — perspective-taking — as a maximally different cognitive operation from arithmetic computation.

---

## Phase VII: Domain Transfer (EXP-20)

*The research program pivots from validating known signatures to testing whether the geometric vocabulary can characterise qualitatively different cognitive operations. Arithmetic reasoning involves computation; empathy involves agent-modelling. If trajectory geometry captures both, it is a general tool for characterising computational modes, not merely a reasoning-specific diagnostic.*

---

### EXP-20: Empathy Geometry

**Date:** March 2026 | **Verdict:** Refuted

**Research Question:** Does perspective-taking (cognitive empathy) produce geometrically distinct hidden-state trajectories compared to matched non-perspective-taking tasks? Does the 57-metric suite detect empathy-specific signatures constituting a third computational mode?

**Prior State:** EXP-19B confirmed the arithmetic signatures are architecturally invariant and that PCR reveals stronger signal than raw metrics suggest. The established geometric vocabulary — radius of gyration, commitment sharpness, directional consistency, effective dimension — was validated across three architectures. The question was whether this vocabulary extends to non-arithmetic cognition.

**Design:**
- **Battery:** 220 prompts — 110 empathy (five categories) and 110 matched control (structurally isomorphic tasks without agent-modelling)
- **Categories:** A (First-Order Belief), B (Second-Order Belief), C (Emotion Recognition), D (Strategic Empathy), E (Perspective Clash)
- **Model:** Qwen2.5-0.5B-Instruct (24 layers, 896 hidden dim)
- **Hidden States:** (24, 48, 896) per problem — 16 prompt tokens + 32 generated tokens at all layers
- **Methodology:** Inductive-first — structure discovery (UMAP, contrastive vectors) before metric imposition, followed by confirmatory factorial disambiguation

**Phase 1 — Inductive Analysis:**
- UMAP manifold analysis (Layer 20, Window 16–40) identified 8 clusters. Empathy trajectories occupied distinct topological regions for categories B, C, and D
- **Empathy Hub:** Cluster 1 (15 trajectories, 100% empathy purity, 100% accuracy) — appeared to be a specialised attractor state for correct perspective-taking
- A contrastive "empathy vector" (896-dim) cleanly separated conditions, with divergence beginning ~5 tokens into generation (Token 21–25) and plateauing by Token ~25
- Geometric separation tracked with accuracy differential — where empathy and control performance diverged most, trajectories also separated most

**Phase 2 — Metric Suite Results:**

| Metric | Cohen's $d$ | $p$-value | Emp Mean | Ctrl Mean |
|--------|------------|-----------|----------|-----------|
| `commitment_sharpness` | **0.77** | 0.000 | 0.380 | 0.287 |
| `vel_autocorr_lag2` | **0.75** | 0.000 | 0.576 | 0.370 |
| `vel_autocorr_lag1` | **0.66** | 0.000 | 0.709 | 0.531 |
| `cos_slope` | 0.32 | 0.025 | 0.020 | 0.016 |
| `radius_of_gyration` | 0.32 | 0.015 | 4.96 | 4.20 |
| `dist_slope` | −0.30 | 0.015 | −0.694 | −0.563 |

- **Phase Specificity Index (PSI):** Global average 0.80 — 337 metrics concentrated in Think window (PSI > 0.67), only 11 in Speak. Effects are cognitive, not language-generation artefacts
- Effect sizes (|$d$| ≈ 0.3–0.8) fall between arithmetic regime effects (|$d$| = 6–8) and within-regime success effects (|$d$| = 0.5–2.2) — a weaker but real geometric signal

**Behavioural Accuracy:**

| Category | Empathy Acc | Control Acc | Gap |
|----------|------------|------------|-----|
| A: First-Order Belief | 43% | 53% | −10% |
| B: Second-Order Belief | **90%** | 5% | **+85%** |
| C: Emotion Recognition | **88%** | 32% | **+56%** |
| D: Strategic Empathy | 95% | 90% | +5% |
| E: Perspective Clash | 0% | 33% | −33% |

Categories B and C showed dramatic empathy advantages, suggesting the model has latent specialised structures for social reasoning. Category E (0% empathy accuracy) indicated the 0.5B model lacks recursive depth for multi-agent perspective clash.

**Phase 4 — Factorial Disambiguation (The Decisive Test):**

A 2×2 factorial design crossed Perspective-Taking (PT vs Objective) with Prose Domain (Narrative vs Technical). ANOVA decomposition across 21 significant metrics at Layer 13:

- **Prose Domain explains 50.6%** of geometric variance on average
- **Perspective-Taking explains 32.9%** of geometric variance
- Of the 8 persistent metrics from Phase 3B:
  - **Genuine empathy metrics (2):** `dist_slope`, `commitment_sharpness`
  - **Prose domain metrics (2):** `radius_of_gyration`, `vel_autocorr_lag1`
  - **Mixed metrics (3):** `drift_to_spread`, `tortuosity`, `cos_to_running_mean`
  - **Null metric (1):** `spectral_entropy`

**Interpretation — Refutation:**
1. **The geometric signal is real but misattributed.** Phases 1–3 detected genuine geometric structure, but the dominant driver was prose domain (narrative vs. technical framing), not perspective-taking per se.
2. **Two metrics survived as genuine empathy signals.** `dist_slope` and `commitment_sharpness` showed significant perspective-taking effects even after controlling for prose domain — suggesting a weak but real empathy-specific geometric component.
3. **`commitment_sharpness` is domain-general.** It appears as a significant predictor in both arithmetic reasoning (EXP-19, where it marks the "Snap" phenomenon) and empathy tasks — it may be a universal signal of computational commitment rather than domain-specific.
4. **The inductive-first methodology was vindicated.** It correctly identified structure that *looked* like empathy geometry; the confirmatory Phase 4 correctly revealed the confound. Without Phase 4, the empathy manifold and empathy vector findings would have been published as genuine.
5. **Factorial disambiguation is now a required step** for any future domain-transfer experiment where task types differ on multiple dimensions.

---

## Methodology Evolution Table

| Decision Point | What Failed | What Replaced It | Experiment |
|---------------|-------------|------------------|------------|
| API output embeddings | Topic and lexical noise dominated operator signal | Open-weights hidden state extraction | EXP-01→02 |
| Scalar magnitude metrics ($||v||$) | Direction discarded; all operators identical | Vector-valued trajectory analysis | EXP-02→08 |
| Static centroid analysis | Operators overlap (F1 ~0.30); position too fuzzy | Dynamic regime classification (Speed, Curvature) | EXP-07→08 |
| Cross-regime probe (balanced dataset) | Topic confounding produced false 100% accuracy | LOPO/LOCO cross-validation; then full dynamic pivot | EXP-03→06→07 |
| Universal success model | 10/14 metrics flip sign across regimes | Regime-conditional monitoring | EXP-14 |
| Fixed 32-token analysis window | Missed late-stage dynamics and commitment | Full trajectory capture with controlled truncation | EXP-14→16B |
| Naive output parsing | Hallucination contamination; 0% Pythia accuracy → parsing artifact | Boundary-aware truncation pipeline | EXP-16 salvage |
| Cross-model replication (TinyLlama-1.1B) | Model below capability floor (0% accuracy) | Architecture-matched capability verification | EXP-09B→16B→19 |
| Introspective self-monitoring | $r \approx 0.00$ with objective geometry | External geometric measurement only | EXP-10 |
| Exploratory metric growth | Inconsistent implementations, no standard | Formalized `TrajectoryMetrics` class (54 metrics) | EXP-18 |
| Multi-architecture using stale hidden states | Pythia-70m data corruption (1,536 vs 512 dim) | Pre-flight tensor shape validation; fresh collection | EXP-18B→19 |
| Cross-process global scope imports | Windows spawn OOM on heavy torch imports | Lazy imports; process isolation | EXP-18B→19 |
| Raw AUC estimates | Attenuation bias from short-sequence noise | Probability Cloud Regression (PCR) denoising | EXP-19B |
| Empathy-control comparison | Prose domain confound dominated perspective-taking signal | Factorial disambiguation (2×2 ANOVA) before attribution | EXP-20 |

---

## Established Knowledge (EXP-01 through EXP-20)

### Core Empirical Findings

1. **Dimensional Collapse in Failure** (EXP-11, 12): Failed direct reasoning collapses into ~3 effective dimensions. Successful CoT reasoning uses ~13. Cohen's $d > 4.5$. Replicated at scale.
2. **Regime-Relative Success Geometry** (EXP-14): 10 of 14 metrics flip sign between CoT and Direct success. No universal "good trajectory" exists; success is regime-dependent.
3. **Difficulty-Driven Geometric Expansion** (EXP-15): $R_g$ effect size scales from $d \approx 5.0$ (Small) to $d > 17.0$ (XL) — resource-rational allocation of geometric volume to problem entropy.
4. **Failure Subtypes** (EXP-13, 19B): Collapsed failures (never engaged reasoning) and Wandering failures (engaged but not committed). G3 failures enter the CoT manifold but fail to commit to the Success Centroid.
5. **Commitment Timing** (EXP-12, 14): Direct answers commit at ~5 tokens; CoT at ~11 tokens. Measurable phase transition (Dimension Drop / "The Snap") is architecturally universal.
6. **Architecture-Invariant Signatures** (EXP-19): 19 signatures hold across Qwen and Pythia families, 410M–1.5B. Phase Count ($d=31.66$), Radius of Gyration ($d=13.99$), Effective Dimension ($d=12.01$).
7. **Regime-Quality Decomposition** (EXP-19B): Regime explains ~80–85% of geometric variance; quality effect (η² ≈ 0.10) is robust within regime (within-CoT AUC 0.78).
8. **PCR-Corrected Signal Strength** (EXP-19B): True predictability likely >0.85; proto-attractors form as early as Layer 0 but are obscured by early-layer noise in raw measurements.
9. **Domain Transfer: Prose Confound** (EXP-20): Geometric separation between empathy and control tasks is primarily driven by prose domain (50.6% of variance) rather than perspective-taking (32.9%). Only `dist_slope` and `commitment_sharpness` survived factorial disambiguation as genuine empathy metrics.
10. **`commitment_sharpness` as Domain-General Signal** (EXP-19, 20): Appears in both arithmetic ("The Snap") and empathy contexts, suggesting it may be a universal computational commitment marker rather than domain-specific.

### Confirmed Null Results

- Static operator vectors are too fuzzy to be useful (F1 ~0.30 maximum)
- Cue-word triggering does not reliably shift geometric regimes (EXP-08')
- Model self-reports do not correlate with internal geometry ($r \approx 0.00$, EXP-10)
- Universal success signatures do not exist (EXP-14)
- Models below capability floor have no reasoning geometry to measure (EXP-09B)
- Empathy-control geometric separation is primarily a prose domain artefact, not perspective-taking (EXP-20)

### Open Questions as of EXP-20

- Do these signatures generalize beyond arithmetic to multi-hop reasoning, ambiguous questions, or retrieval tasks?
- Do they persist at frontier scale (>10B parameters)?
- Can intervening on trajectory geometry (activation patching) causally redirect G3-bound trajectories toward the Success Attractor?
- Does the full PCR-corrected metric suite replicate on non-Qwen, non-Pythia architectures (Llama-3, Gemma)?
- Can empathy-specific geometric signatures be isolated from prose domain effects with properly matched factorial designs at larger model scale?
- Is `commitment_sharpness` a domain-general marker of computational commitment? Does it appear in other cognitive operations (e.g., retrieval, planning)?

---

---

## Phase VIII — Cross-Architecture Deepening and Causal Validation

### EXP-21: Gemma Deep Dive (March 2026) — SUCCESS WITH MECHANISTIC CONSTRAINTS

**Research Question:** Do the trajectory-geometry findings replicate on Gemma 3 1B, and can GemmaScope 2 reveal a sparse feature-level mechanism underlying The Snap and the Success Attractor?

**Key Results:**
- **H1 (Replication): Confirmed.** Regime relativity replicated cleanly on Gemma 3 1B. CoT trajectories are geometrically distinct from Direct, and geometry predicts success within regime.
- **H2 (Proto-Attractor): Partially confirmed.** PCR gains on Gemma are positive but smaller than Qwen — strongest for Direct Layer 0 (gain +0.068, CI excluding zero), CoT Layer 0 modest (+0.031, CI overlapping zero).
- **H3 (Sparse Circuit): Not confirmed.** The negative SAE-PCR result is real (not a calibration artefact). Aggregate SAE statistics provide small complementary signal in deeper layers but do not rescue the main mechanistic claim. Individual SAE features do not achieve d > 1.5 for reasoning discrimination.
- **H4 (Architectural Fingerprint): Confirmed.** Gemma's 5:1 local/global attention schedule produces measurable geometric discontinuities at global attention layers.

**Impact:** Arithmetic PCR claims strengthened across a third architecture family (Gemma joins Qwen and Pythia). Generalization and mechanistic claims must be constrained. The SAE feature space has rich latent structure but no single factor cleanly captures "reasoning success."

**Bridge to EXP-22:** The weak factor-level discrimination motivated a pivot from unsupervised factor-based ablation targets to supervised feature selection using partial correlation filtering. The question became: can individual SAE features, selected by their partial correlation with correctness controlling for difficulty, causally affect construct-specific performance when ablated?

### EXP-22: Construct-Specific SAE Feature Ablation (March 2026) — SUCCESS

**Research Question:** Can we identify SAE features that causally implement carry-operation processing, and demonstrate construct-specific performance decline when those features are ablated?

**Motivation:** The Trajectory Geometry programme established that geometric signatures reliably track reasoning quality, but the evidence was purely observational. EXP-22 attempts the causal step: if geometric signatures reflect real computation, then ablating the features that underlie that computation should produce measurable, construct-specific performance decline. Success here would provide the first piece of a Goodhart-resistant evaluation framework.

**Phase 1 — Factor Analysis (COMPLETE):**
- 80 features passed two-stage filtering from 16,384 GemmaScope SAE features
- 9-factor solution (KMO=0.908, 72.5% variance explained)
- F1 (Content/Difficulty, 24.7% variance) strongly correlated with difficulty bin (r=+0.374) and digit count (r=+0.387)
- F9 (Control candidate, 4.1% variance) showed d=+0.482 for G4 vs G3 — best discriminator but below d=1.5 threshold
- Strategic pivot: from factor-based to supervised feature selection

**Phase 2 — Supervised Feature Selection (COMPLETE):**
- Partial correlation filtering: 80 features ranked by |partial r| with correctness controlling for difficulty bin and digit count
- All top 30 features significant at p<0.001
- 9 ablation candidates survived (not F1-content, significant partial correlation with correctness)
- Feature classification by correlation profile (Neuronpedia does not index Gemma 3 / GemmaScope 2):
  - Feature 6340: CONTROL (d=+0.669, clean discriminator, no content correlation)
  - 8 others classified as MIXED (discriminative but also content-correlated)
- Ablation set (sorted by |d|): 6340, 14452, 7994, 2126, 4596, 5078, 1726, 10078, 2128
- Content control set (F1 top 10): 379, 1258, 7722, 12671, 4324, 8279, 2028, 6353, 2392, 594

**Phase 3 — Assessment Battery (COMPLETE):**
- 80-problem battery: 40 carry (target), 20 no-carry (control), 20 formatting (control)
- Carry subtypes: single-carry addition, double-carry addition, carry-with-multiplication, nested operations

**Phase 4 — Ablation Experiment (COMPLETE, 14/14 conditions, 159 minutes):**

Critical implementation discovery: the Gemma3DecoderLayer (with transformers output_capturing wrapper) returns a plain Tensor `(batch, seq_len, d_model)`, not a tuple. This caused the initial ablation hook to silently fail. Additionally, full SAE reconstruction introduces massive error (mean diff ~1000 per dimension), so the experiment uses **residual ablation**: subtracting only the contribution of the targeted features from the hidden state, preserving the rest of the representation. A 224x speedup was achieved by pre-slicing encoder weights to only target features.

**Final results (14/14 conditions):**

| Condition | Carry Acc | No-Carry Acc | Fmt Acc | Carry Drop | Specificity |
|-----------|-----------|--------------|---------|------------|-------------|
| Baseline | 75.0% | 100.0% | 75.0% | — | — |
| Top 5, s=0.25 | **60.0%** | 100.0% | 75.0% | **15pp** | **infinite** |
| Top 5, s=0.50 | 2.5% | 55.0% | 70.0% | 72.5pp | 1.6:1 |
| Top 5, s=1.00 | 0.0% | 0.0% | 45.0% | 75pp | <1 |
| All 9, s=0.25 | **45.0%** | 100.0% | 70.0% | **30pp** | **infinite** |
| All 9, s=0.50 | 0.0% | 0.0% | 60.0% | 75pp | <1 |
| All 9, s=1.00 | 0.0% | 0.0% | 0.0% | 75pp | <1 |
| Random ctrl, s=0.50 | 0.0% | 15.0% | 25.0% | 75pp | 0.9:1 |
| Content ctrl, s=0.50 | 30.0% | 50.0% | 75.0% | 45pp | 0.9:1 |
| High-freq ctrl, s=0.50 | 0.0% | 0.0% | 0.0% | 75pp | <1 |
| Commitment-only, s=0.50 | 25.0% | 85.0% | 65.0% | 50pp | **3.3:1** |
| Single 6340, s=0.50 | 70.0% | 100.0% | 75.0% | **5pp** | **infinite** |
| Single 14452, s=0.50 | 67.5% | 100.0% | 85.0% | **7.5pp** | **infinite** |
| Single 7994, s=0.50 | 67.5% | 100.0% | 55.0% | **7.5pp** | **infinite** |

**The headline results:**

1. **Construct specificity at s=0.25 is real and strong.** Top 5 features produce a 15pp carry-specific drop with literally zero impact on no-carry (100% -> 100%) or formatting (75% -> 75%). All 9 features produce a 30pp drop with no-carry still at 100%. Both conditions achieve effectively infinite specificity ratios.

2. **Single-feature ablations are the strongest evidence.** Each individual feature (6340, 14452, 7994) at s=0.50 produces a 5-7.5pp carry drop with zero no-carry impact — at the same dose that causes total destruction when applied to 9 random features. This demonstrates that individual supervised features carry genuine construct-specific causal weight.

3. **Control dissociation validates the feature selection pipeline.** The three control sets show qualitatively different disruption patterns:
   - **Random control:** Uniform destruction across all categories (s=0.50: carry 0%, no-carry 15%, formatting 25%)
   - **Content control:** Arithmetic-broad disruption but formatting-sparing (carry 30%, no-carry 50%, formatting 75%)
   - **High-frequency control:** Total destruction (0% across the board)
   - **Targeted construct features:** Carry-preferential disruption with control-task preservation

4. **Dose-response follows pharmacological dynamics.** Scale 0.25 is the "therapeutic dose" (construct-specific effects, no side effects). Scale 0.50 is "overdose" (large effects but collateral damage). Scale 1.00 is "lethal dose" (everything fails). The therapeutic index is approximately 2x.

5. **Token-restricted ablation preserves specificity at higher doses.** Commitment-only ablation at s=0.50 achieves 3.3:1 specificity vs 1.6:1 for all-token at the same dose, confirming that carry computation is distributed but concentrated at commitment points.

**Phase 4b — Extended Random Controls (COMPLETE, 12 additional conditions):**

The initial Phase 4 results were strengthened by two rounds of additional controls:

*Multi-feature random controls at s=0.25 (6 seeds):* Six independent random draws of 9 activation-frequency-matched features, all ablated at the therapeutic dose. The noise distribution was much higher than expected: mean carry drop +22.1pp (SD 22.0pp), no-carry drop mean +15.0pp (SD 16.7pp). Three of six random draws preserved no-carry at 100%. This means the multi-feature s=0.25 result, while real, sits within a noisy random distribution rather than cleanly above it.

*Single-feature random controls at s=0.50 (6 seeds):* **This is the definitive test.** Six activation-frequency-matched random features, each ablated alone at s=0.50.

| Condition | Carry Drop | No-Carry Drop | Specificity |
|-----------|------------|---------------|-------------|
| **Targeted 6340** | **+5.0pp** | **0.0pp** | **infinite** |
| **Targeted 14452** | **+7.5pp** | **0.0pp** | **infinite** |
| **Targeted 7994** | **+7.5pp** | **0.0pp** | **infinite** |
| Random mean (n=6) | +20.4pp | +17.5pp | ~1.1:1 |
| Random SD | 18.1pp | 18.9pp | — |
| Random range | [-7.5, +47.5] | [0.0, +50.0] | — |

The result was more informative than predicted. Random single features at s=0.50 are **far more destructive** than targeted features — they cause large, non-specific damage (mean 20pp carry AND 18pp no-carry). Targeted features cause **small, perfectly specific** damage (5-7.5pp carry, exactly 0pp no-carry). The targeted features are scalpels in a distribution of wrecking balls. This pharmacological selectivity pattern is the strongest evidence for construct specificity in the programme.

**Caveats:** Sample sizes are small (3 targeted, 6 random). Geometric metrics under ablation were not computed. Multi-feature s=0.25 noise floor is high — lead with single-feature evidence.

**Bridge to the Goodhart Problem:** EXP-22's construct-specific ablation test provides a mechanistic validator that goes beyond accuracy. A model achieving high carry accuracy via genuine computation should (a) activate the carry features during processing, and (b) suffer carry-specific degradation when those features are ablated. A model achieving accuracy via memorization or pattern-matching shortcuts should show neither. This establishes the first piece of a dual-channel Goodhart-resistant evaluation framework (geometry + feature ablation). EXP-23 is designed to test this directly by fine-tuning a model and checking whether the validator can distinguish genuine reasoning from shortcut-learning.

---

## Phase IX — The Goodhart-Resistant Evaluation Framework

The convergence of EXP-19/19B (geometric signatures), EXP-21 (cross-architecture replication + SAE feature structure), and EXP-22 (causal feature ablation) brings the programme to its central theoretical payoff: a framework for validating whether a language model is genuinely reasoning or merely retrieving.

### The Goodhart Problem in AI Evaluation

Goodhart's Law states that "when a measure becomes a target, it ceases to be a good measure." In the context of LLM evaluation, accuracy on benchmarks is the standard measure of reasoning capability. But models can achieve high accuracy through:

- **Genuine computation (G4 geometry):** The model uses multi-step internal processing — dimensional expansion, exploration, commitment — to derive the answer. Geometric signatures show the Explore-Commit phase transition and the Success Attractor. Carry-implementing SAE features activate during computation tokens.

- **Shortcut retrieval (G2 geometry):** The model pattern-matches the problem to a memorized or approximated answer. Geometric signatures show low expansion, early commitment, and direct-answer-like trajectories. Carry-implementing SAE features may not activate (the model bypasses the carry mechanism entirely).

Standard accuracy metrics cannot distinguish these two strategies. A model fine-tuned on arithmetic problems may achieve 95% accuracy by memorizing common problem-answer pairs, and this would be indistinguishable from genuine computation by accuracy alone.

### The Dual-Channel Validator

EXP-22 establishes the second channel of a two-channel validation system:

1. **Geometric channel (EXP-01–21):** Measures trajectory shape during inference. Genuine reasoning produces characteristic expansion, commitment timing, and convergence patterns. Architecture-invariant across Qwen, Pythia, and Gemma.

2. **Feature channel (EXP-22):** Measures whether carry-implementing SAE features are causally engaged during processing. The ablation test provides a direct probe: if ablating carry features degrades carry accuracy, the model was using them. If not, the model was bypassing them (likely using shortcuts).

These channels are partially independent — a model could hypothetically show the right geometry while using different features, or activate the right features while following an unusual trajectory. The redundancy makes the validator more robust to gaming.

### What Remains (EXP-23 and Beyond)

The Goodhart proof loop requires:
1. A model that achieves high accuracy (the "target") -- **easy, via fine-tuning**
2. Evidence that the accuracy was achieved via shortcuts -- **requires geometry + ablation testing the fine-tuned model**
3. Demonstration that the validator detects the difference -- **the core EXP-23 test**

If EXP-23 succeeds, the programme will have demonstrated that trajectory geometry and SAE feature ablation, together, constitute a Goodhart-resistant evaluation framework for reasoning: a system where high accuracy is necessary but not sufficient, and where the mechanistic path to that accuracy is independently verifiable.

---

### Research Agenda: Forward Direction

The following experiments represent the highest-value next steps, ordered by priority:

**Immediate (pre-EXP-23 gap-filling):**

1. **Random control at s=0.25.** Fill the critical gap: test random feature ablation at the therapeutic dose to establish the noise floor for construct-specific comparisons.
2. **Geometric shift computation.** Extract hidden-state trajectories during ablated inference and compute commitment_sharpness, D_eff, R_g shifts. This empirically closes the geometric-feature bridge.
3. **Expanded assessment battery.** Double to 160 problems for higher statistical power on single-feature and control comparisons.

**EXP-23: The Goodhart Test (next major experiment):**

4. Fine-tune Gemma 3 1B on 200 training-split arithmetic problems.
5. Test on held-out 200 + the Phase 3 assessment battery.
6. Apply the dual validator:
   - Does the fine-tuned model show G4-like geometry on carry tasks?
   - Does ablating carry features degrade its performance?
7. Compare: genuine-computation fine-tuning vs. shortcut fine-tuning (if achievable by manipulating training data or objective).

**EXP-24+: Generalization and Robustness:**

8. **Multi-layer ablation.** Test whether ablating the same features across layers 8-12 amplifies or changes the construct-specific effect.
9. **Cross-model replication.** Replicate the ablation experiment on other models with available SAEs.
10. **Domain transfer.** Test whether the supervised feature selection + ablation methodology transfers to non-arithmetic domains (e.g., logical reasoning, reading comprehension, code generation).
11. **Continuous construct measurement.** Move from binary ablation (on/off) to continuous feature modulation, enabling measurement of the "reasoning dose-response" across task difficulty levels.

**Theoretical:**

12. **Formalize the dual-channel validator** as a general framework for Goodhart-resistant evaluation of any cognitive construct (not just arithmetic carry). The key ingredients are: (a) a construct-specific assessment battery, (b) geometric trajectory signatures of genuine computation, (c) causal feature ablation tests. If these generalize beyond arithmetic, the framework becomes a general tool for mechanistic evaluation of language model capabilities.

---

### EXP-23: Perturbation Spectroscopy (PS-1, PS-2, PS-3)

**Date:** March 2026 | **Verdict:** Success

**Research Question:** Do specific SAE features exert universally measurable causal influence over reasoning tasks across varying complexities and contextual lengths, and is the geometric representation of these failures consistent?

**Prior State:** EXP-22 proved that specific features represent targeted semantic actions (e.g., carrying in arithmetic), but this was demonstrated on short, synthetically simple math problems. It wasn't known if these features were active in complex reasoning, or if the intervention behaved linearly under prolonged semantic strain.

### Phase VIII: Refined Spectroscopy & Geometric Profiling - 2026-03-27

Transitioned to a more robust perturbation screen (alpha range [-0.25, 1.0]) at Layer 10 (GemmaScope SAE). Key findings:
- **Infrastructure Bottleneck**: Characterized Feature 869 as a "logic-agnostic" infrastructure node (activation precedes logic commitment) rather than a carry-specific node.
- **Geometric Profiling**: Successfully clustered perturbation vectors into "Bending" and "Shattering" failure modes, providing a geometric signature of model failure.
- **Causal Agonists**: Identified Feature 212 as a potential "carry-agonist" (amplification improves performance), a rare find in SAE feature space.

### Phase I: Robustness & Refinement (The ArXiv Gate) - 2026-03-28

Completed Phase I of the research roadmap, achieving several critical validation milestones:
- **Basis-Invariance of $t_c$**: Demonstrated that geometric commitment point ($t_c$) is identical (Corr=1.0) whether measured in the Residual Stream (Raw), PCA projections, or SAE activation space. This establishes $t_c$ as a fundamental property of the computation.
- **Causal Agonist Verification**: Verified the "agonist" property of Feature 212 at Layer 10. Hook monitoring confirmed that the feature is reliably active during carry-logic problems and its amplification correlates with successful task completion.
- **Scale and Robustness**: Analysis of ~1,350 trajectories across 50 batches confirms the stability of these findings across the entire Trajectory Geometry programme.

**Findings**: Causal-geometric link is robust. $t_c$ is an invariant. Agonist features are real and predictable. Ready for functional decomposition.

**Method:** 
- **Model:** Gemma 3 1B Instruct (SAE Layer 10)
- **Dataset:** The 80-question "Cognitive Decathlon" (a rigorous logic and math battery).
- **Procedure:** Spectral causal ablation across 80 designated logic/math features. Intervention was tested at `alpha=0.25` (selective) and `alpha=0.5` (catastrophic stress-test). Ablated states were stored and graded deterministically.

**Results:**
- **Infrastructure Bottlenecks:** Identified Feature 869 as a high-frequency (87%) structural node. Its 36% accuracy collapse at `alpha=0.5` represents non-specific structural failure rather than a "logic scalpel."
- **Causal Decoupling (Feature 6340):** The star feature of EXP-22 registered a **0.0% delta** on the complex logic tasks. The model does not use simple arithmetic circuits for math embedded inside complex text puzzles; it shifts to a disjoint processing regime (Dynamic Circuit Reconfiguration).
- **Arithmetic Specialized Gate (Feature 212):** Confirmed specific causality on mathematical sub-tasks (-17.5% Acc) while remaining inert on linguistic logic.

**Interpretation — success:** EXP-23 mapped the structure of reasoning execution and produced empirical proof that models do not use the same static circuitry for simple and complex versions of the same abstract problem. The re-characterization of 869 as infrastructure provides a vital "noise floor" for geometric analysis.

**Bridge to PS-4:** Knowing *which* features represent infrastructure (869) vs. specialized logic (212) allows us to analyze the geometric shape of selective vs. global breakage. Phase PS-4 (Geometric Profiling) has successfully extracted these signatures.

---

*For the empirical fact layer, see [findings-catalogue.md](../reference/findings-catalogue.md).*
*For complete metric definitions, see [metrics-appendix.md](../reference/metrics-appendix.md) and [metric-definitions.md](../reference/metric-definitions.md).*
*For PCR analysis data, see `experiments/EXP-19_Robustness_2026-02-14/data/analysis_19b/`.*
*For empathy geometry results, see `experiments/EXP-20_EmpathyGeometry_2026-03-13/results/`.*
*For EXP-22 ablation results and full Phase 4 report, see `experiments/EXP-22/results/exp22_phase4_report.md`.*
*For EXP-23 perturbation results and PS-3 synthesis, see `experiments/EXP-23_PerturbationSpectroscopy/ps3/ps3_synthesis_report.md`.*


