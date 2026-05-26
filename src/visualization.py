"""
visualization.py — Plotting and visualization for BioLens interpretability results.

Generates publication-quality figures for:
- Layer-wise probe accuracy curves
- Activation patching heatmaps
- Safety direction projections
- Feature clustering visualizations
- Concept direction similarity matrices
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from typing import Optional
from pathlib import Path


# Set publication style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Color palette inspired by scientific journals
COLORS = {
    "primary": "#2563EB",
    "secondary": "#DC2626",
    "tertiary": "#059669",
    "quaternary": "#D97706",
    "benign": "#3B82F6",
    "sensitive": "#EF4444",
    "neutral": "#6B7280",
}


def plot_probe_accuracy_by_layer(
    results_by_concept: dict[str, list],
    save_path: Optional[str] = None,
    title: str = "Linear Probe Accuracy Across Layers",
) -> plt.Figure:
    """
    Plot probe accuracy as a function of layer depth for multiple concepts.

    This is THE key figure for the project: it shows at which layer
    the model transitions from surface patterns to biological abstraction.

    Args:
        results_by_concept: Dict mapping concept names to lists of ProbeResults.
        save_path: Path to save the figure.
        title: Figure title.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = list(COLORS.values())
    for i, (concept, results) in enumerate(results_by_concept.items()):
        layers = [int(r.layer.split("_")[1]) for r in results]
        means = [r.cv_mean for r in results]
        stds = [r.cv_std for r in results]

        color = colors[i % len(colors)]
        ax.plot(layers, means, "o-", color=color, label=concept, linewidth=2, markersize=5)
        ax.fill_between(
            layers,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            alpha=0.15,
            color=color,
        )

    ax.axhline(y=0.5, color=COLORS["neutral"], linestyle="--", alpha=0.5, label="Chance")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cross-Validated Accuracy")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_ylim(0.3, 1.05)
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path)
        print(f"Saved to {save_path}")

    return fig


def plot_patching_heatmap(
    effect_sizes: dict[str, float],
    n_layers: int,
    save_path: Optional[str] = None,
    title: str = "Activation Patching: Causal Effect by Component",
) -> plt.Figure:
    """
    Heatmap showing causal effect of patching each layer/component.

    Rows = layers, columns = component types (attention, MLP, residual).
    Brighter = more causally important for the prediction.
    """
    # Parse component names into layer x component matrix
    components = set()
    for key in effect_sizes:
        parts = key.split("_")
        comp = "_".join(parts[2:]) if len(parts) > 2 else "resid"
        components.add(comp)

    components = sorted(components)
    matrix = np.zeros((n_layers, len(components)))

    for key, effect in effect_sizes.items():
        parts = key.split("_")
        layer = int(parts[1])
        comp = "_".join(parts[2:]) if len(parts) > 2 else "resid"
        col = components.index(comp)
        matrix[layer, col] = effect

    fig, ax = plt.subplots(figsize=(max(4, len(components) * 1.5), max(6, n_layers * 0.4)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlBu_r", vmin=-0.1, vmax=1.0)

    ax.set_xticks(range(len(components)))
    ax.set_xticklabels([c.replace("_", "\n") for c in components])
    ax.set_ylabel("Layer")
    ax.set_title(title)

    plt.colorbar(im, ax=ax, label="Fraction of total effect recovered")

    if save_path:
        fig.savefig(save_path)

    return fig


def plot_safety_projections(
    benign_projections: np.ndarray,
    sensitive_projections: np.ndarray,
    threshold: float,
    save_path: Optional[str] = None,
    title: str = "Safety Direction Projections",
) -> plt.Figure:
    """
    Distribution plot showing how benign and sensitive prompts project
    onto the learned safety direction.

    Good separation = the direction is a useful internal monitor.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.hist(benign_projections, bins=30, alpha=0.6, color=COLORS["benign"],
            label="Benign", density=True, edgecolor="white", linewidth=0.5)
    ax.hist(sensitive_projections, bins=30, alpha=0.6, color=COLORS["sensitive"],
            label="Sensitive", density=True, edgecolor="white", linewidth=0.5)

    ax.axvline(x=threshold, color=COLORS["neutral"], linestyle="--",
               linewidth=2, label=f"Threshold ({threshold:.2f})")

    ax.set_xlabel("Projection onto Safety Direction")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.2)

    if save_path:
        fig.savefig(save_path)

    return fig


def plot_concept_similarity_matrix(
    concept_directions: dict[str, np.ndarray],
    save_path: Optional[str] = None,
    title: str = "Concept Direction Similarity",
) -> plt.Figure:
    """
    Heatmap of cosine similarities between all pairs of concept directions.

    Reveals which biological concepts the model entangles vs. separates.
    High similarity between "toxicity" and "safety_sensitive" would be
    particularly interesting — it would mean the model's internal
    representation of toxicity partly overlaps with its representation
    of dual-use knowledge.
    """
    names = list(concept_directions.keys())
    n = len(names)
    sim_matrix = np.zeros((n, n))

    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            d_i = concept_directions[name_i]
            d_j = concept_directions[name_j]
            sim_matrix[i, j] = np.dot(d_i, d_j) / (
                np.linalg.norm(d_i) * np.linalg.norm(d_j)
            )

    fig, ax = plt.subplots(figsize=(8, 7))
    mask = np.zeros_like(sim_matrix, dtype=bool)
    np.fill_diagonal(mask, True)

    sns.heatmap(
        sim_matrix,
        xticklabels=names,
        yticklabels=names,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        mask=mask,
        ax=ax,
        square=True,
    )
    ax.set_title(title)

    if save_path:
        fig.savefig(save_path)

    return fig


def plot_layer_auroc_curve(
    layer_aurocs: dict[str, float],
    save_path: Optional[str] = None,
    title: str = "Safety Direction AUROC by Layer",
) -> plt.Figure:
    """
    Plot AUROC of the safety direction classifier across layers.

    Shows at which depth the model best separates benign from sensitive content.
    """
    layers = sorted(layer_aurocs.keys(), key=lambda x: int(x.split("_")[1]))
    layer_nums = [int(l.split("_")[1]) for l in layers]
    aurocs = [layer_aurocs[l] for l in layers]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(layer_nums, aurocs, "o-", color=COLORS["secondary"], linewidth=2, markersize=6)
    ax.axhline(y=0.5, color=COLORS["neutral"], linestyle="--", alpha=0.5, label="Chance")
    ax.axhline(y=0.8, color=COLORS["tertiary"], linestyle=":", alpha=0.5, label="Good (0.8)")

    best_layer = max(layer_aurocs, key=layer_aurocs.get)
    best_idx = layers.index(best_layer)
    ax.annotate(
        f"Best: {aurocs[best_idx]:.3f}",
        xy=(layer_nums[best_idx], aurocs[best_idx]),
        xytext=(10, 15),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=COLORS["secondary"]),
    )

    ax.set_xlabel("Layer")
    ax.set_ylabel("AUROC")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.05)

    if save_path:
        fig.savefig(save_path)

    return fig


def create_summary_figure(
    probe_results: dict,
    patching_effects: dict,
    safety_result,
    n_layers: int,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create a multi-panel summary figure combining all key results.

    Panel A: Probe accuracy by layer
    Panel B: Patching heatmap
    Panel C: Safety direction projections
    Panel D: Concept similarity matrix
    """
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # This is a template — fill in with actual results
    fig.suptitle("BioLens: Mechanistic Interpretability of Biomedical LMs",
                 fontsize=14, fontweight="bold", y=0.98)

    for i, (label, title) in enumerate([
        ("A", "Probe Accuracy by Layer"),
        ("B", "Activation Patching Effects"),
        ("C", "Safety Direction Separation"),
        ("D", "Concept Direction Similarity"),
    ]):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        ax.set_title(f"{label}. {title}", loc="left", fontweight="bold")
        ax.text(0.5, 0.5, f"[{title}]", ha="center", va="center",
                fontsize=12, color=COLORS["neutral"], transform=ax.transAxes)

    if save_path:
        fig.savefig(save_path)

    return fig
