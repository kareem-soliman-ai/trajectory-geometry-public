# Trajectory Geometry: Regime-Dependent Geometric Signatures in Transformer Hidden State Trajectories

**Paper:** ArXiv link to be added after submission  
**Author:** Kareem Soliman (`kareem.soliman@outlook.com.au`)  
**Status:** Pre-print under review

## Summary

This repository contains the data, analysis code, figures, and reproduction notebooks for
*The Shape of Reasoning*, which introduces trajectory geometry as a framework for analysing
hidden-state dynamics in transformer language models during multi-step reasoning.

The central empirical result is the **Regime-Relativity Principle**:
there is no universal "good trajectory" for reasoning. Geometric signatures of success
reverse sign between Direct-answer and Chain-of-Thought (CoT) regimes.

For the audited March 2026 Qwen-0.5B manuscript pass included here:

- Mean regime `eta^2` across all metrics at peak layers (10-16): `27.78%`
- Regime share of explained variance for the six most regime-sensitive metrics at layers 10-16: `74.58%`
- Basis robustness of commitment timing on Gemma 3 1B: raw vs PCA-16 `r = 0.918`, raw vs SAE-16k `r = 0.576`

These values supersede older draft phrasing that described regime as explaining `80--85%`
of total geometric variance.

## Quick Start

```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_dimensional_collapse.ipynb
```

## Repository Structure

- `paper/` - audited LaTeX manuscript used for the March 2026 arXiv submission pass
- `data/` - datasets organised by model family and derived analysis outputs
- `src/` - cleaned core analysis code for metrics, PCR, hidden-state extraction, and statistics
- `notebooks/` - reproduction notebooks, one per core finding
- `figures/` - audited publication figures plus supporting audit reports
- `docs/` - research narrative, findings catalogue, metric definitions, and experiment progression

## Models Used

| Model | Family | Parameters | Layers | Hidden Dim |
| --- | --- | --- | --- | --- |
| Qwen2.5-0.5B-Instruct | Qwen | 0.5B | 24 | 896 |
| Qwen2.5-1.5B-Instruct | Qwen | 1.5B | 28 | 1536 |
| EleutherAI/pythia-410m | GPT-NeoX | 410M | 24 | 1024 |
| google/gemma-3-1b-it | Gemma | 1B | 26 | 1152 |

## Hardware

The original experiments were conducted on consumer hardware, primarily:

- AMD RX 5700 XT (8GB VRAM, DirectML acceleration)

No specialised cluster infrastructure was required for the core Qwen-0.5B analyses.

## Reproduction Guide

The easiest way to audit the paper-facing findings is:

1. Start with `notebooks/01_dimensional_collapse.ipynb` through `notebooks/09_pcr_denoising.ipynb`.
2. Inspect the copied source tables in `data/analysis/` and `data/gemma3_1b/`.
3. Compare the resulting figures and statistics to the audited outputs under `figures/`.

## Citation

```bibtex
@article{soliman2026trajectory,
  title={The Shape of Reasoning: Regime-Dependent Geometric Signatures in Transformer Hidden State Trajectories},
  author={Soliman, Kareem},
  journal={arXiv preprint},
  year={2026}
}
```

## AI Disclosure

AI tools (Claude, Anthropic; Gemini, Google) were used for implementation support during
analysis development. All experimental design, theoretical contributions, and interpretive
frameworks are the author's own.

## License

MIT License
