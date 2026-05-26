# 🧬 BioLens: Mechanistic Interpretability for Biomedical Language Models

**How do language models represent biological knowledge — and can we detect when that knowledge becomes dangerous?**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-preprint-b31b1b.svg)](#) <!-- Update when posted -->

---

## Motivation

Large language models trained on biomedical corpora encode rich representations of drug mechanisms, protein interactions, and metabolic pathways. But **what do these representations actually look like inside the model?** And critically — can we build tools to detect when a model is activating knowledge that poses dual-use biosecurity risk?

This project applies **mechanistic interpretability** techniques — originally developed for general-purpose LLMs ([Elhage et al., 2022](https://transformer-circuits.pub/2022/toy_model/index.html); [Bricken et al., 2023](https://transformer-circuits.pub/2023/monosemantic-features/index.html)) — to biomedical transformer models. The goal is to:

1. **Map how biomedical models represent drug-target interactions** at the level of individual neurons and features
2. **Identify interpretable "safety-relevant" directions** in activation space (e.g., toxicity, synthesis feasibility, dual-use potential)
3. **Build a lightweight classifier** that flags when a model's internal activations suggest it is reasoning about dangerous biological knowledge

This work sits at the intersection of AI safety and biology — a space where rigorous evaluation requires both ML depth and domain expertise.

## Why This Matters

Anthropic's [Responsible Scaling Policy](https://www.anthropic.com/index/anthropics-responsible-scaling-policy) identifies biological risks as a key evaluation domain for frontier models. Current biological evaluations are mostly behavioral (prompt-based). This project explores whether **internal model representations** can provide an earlier, more robust signal — a "neural biomarker" for dangerous knowledge, analogous to how translational biomarkers predict clinical outcomes before they manifest.

## Project Structure

```
biolens/
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
│
├── notebooks/
│   ├── 01_model_exploration.ipynb        # Load model, inspect architecture, baseline probing
│   ├── 02_activation_extraction.ipynb    # Extract activations for drug-target prompts
│   ├── 03_probing_classifiers.ipynb      # Linear probes for biological concepts
│   ├── 04_activation_patching.ipynb      # Causal tracing: which layers/heads matter?
│   ├── 05_feature_visualization.ipynb    # Visualize discovered features
│   └── 06_safety_directions.ipynb        # Identify and validate safety-relevant directions
│
├── src/
│   ├── __init__.py
│   ├── models.py                # Model loading and hooking utilities
│   ├── activations.py           # Activation extraction and caching
│   ├── probes.py                # Linear probing classifiers
│   ├── patching.py              # Activation patching experiments
│   ├── features.py              # Feature analysis and clustering
│   ├── safety.py                # Safety-direction identification
│   └── visualization.py         # Plotting and interactive dashboards
│
├── data/
│   ├── prompts/
│   │   ├── drug_target_pairs.jsonl       # Curated drug-target interaction prompts
│   │   ├── metabolic_pathways.jsonl      # Pathway completion prompts
│   │   ├── safety_benign.jsonl           # Benign biological queries
│   │   └── safety_sensitive.jsonl        # Dual-use relevant queries (carefully curated)
│   └── processed/
│       └── .gitkeep
│
├── configs/
│   ├── model_config.yaml
│   └── experiment_config.yaml
│
├── tests/
│   ├── test_activations.py
│   ├── test_probes.py
│   └── test_patching.py
│
├── results/
│   └── .gitkeep
│
└── blog/
    └── post_1_bio_interpretability.md    # Companion write-up
```

## Approach

### Phase 1: Representation Mapping (Weeks 1–3)

**Model**: [BioGPT](https://github.com/microsoft/biogpt) or [PubMedBERT](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext) — small enough for interpretability work on a single GPU.

- Extract activations across all layers for curated biomedical prompts
- Train **linear probes** to classify biological concepts from intermediate representations:
  - Drug mechanism of action (kinase inhibitor vs. antimetabolite vs. immunotherapy)
  - Target protein family
  - Toxicity class (from known clinical data)
  - Metabolic pathway membership

**Key question**: At which layer does the model transition from "surface-level token patterns" to "meaningful biological abstraction"?

### Phase 2: Causal Analysis (Weeks 4–5)

- Apply **activation patching** (causal tracing) to identify which attention heads and MLPs are causally responsible for:
  - Correctly predicting drug-target binding
  - Distinguishing on-target vs. off-target effects
  - Representing metabolic pathway flux (drawing on my LC-MS/MS and isotope tracing experience)

- Compare causal circuits for "safe" biological reasoning vs. "sensitive" biological reasoning

### Phase 3: Safety Directions (Weeks 6–8)

- Use **contrastive activation analysis** between benign and sensitive prompts to identify directions in activation space that correlate with dual-use knowledge
- Build a lightweight binary classifier on internal activations that flags sensitive reasoning
- Validate with held-out prompt sets
- Analyze failure modes and limitations

## Technical Stack

- **PyTorch** + **TransformerLens** (for hooking and activation extraction)
- **scikit-learn** (linear probes)
- **Plotly / matplotlib** (visualization)
- **Weights & Biases** (experiment tracking)
- **HuggingFace Transformers** (model loading)

## Getting Started

```bash
git clone https://github.com/spoddar2008/biolens.git
cd biolens
pip install -r requirements.txt

# Run the first notebook
jupyter notebook notebooks/01_model_exploration.ipynb
```

## Key Design Decisions

**Why linear probes?** If a linear probe can decode a concept from activations, that concept is represented as a direction in activation space — the simplest and most interpretable form of representation. This is the standard approach in mechanistic interpretability (Alain & Bengio, 2017).

**Why small models?** Interpretability requires examining every layer and head. BioGPT (347M parameters) is large enough to encode meaningful biomedical knowledge but small enough for exhaustive analysis on accessible hardware.

**Why this matters for safety?** Behavioral evaluations (prompting a model and checking its output) can be gamed. Internal activation monitoring is harder to circumvent and could provide defense-in-depth for biological risk evaluation.

## Author

**Soumya Poddar, PhD**
Principal Scientist, Translational Medicine | Kite Pharma (Gilead)
MS Computer Science (AI) | Georgia Institute of Technology

Background: 22 peer-reviewed publications in cancer biology, 6 patents, experience with IND-enabling studies and clinical trial biomarker strategy. Currently applying mechanistic interpretability techniques to understand how models represent biological knowledge.

- [LinkedIn](https://www.linkedin.com/in/soumyapoddar-2805/)
- [Publications](https://www.ncbi.nlm.nih.gov/myncbi/1hiyil5CaQh5l/bibliography/public/)
- [GitHub](https://github.com/spoddar2008)

## Related Work

- Elhage et al. (2022). "Toy Models of Superposition." Transformer Circuits Thread.
- Bricken et al. (2023). "Towards Monosemanticity." Transformer Circuits Thread.
- Nanda et al. (2023). "Progress measures for grokking via mechanistic interpretability."
- Anthropic (2023). Responsible Scaling Policy.

## License

MIT License. See [LICENSE](LICENSE) for details.

---

*This project is independent research and is not affiliated with or endorsed by Anthropic, Kite Pharma, or Gilead Sciences.*
