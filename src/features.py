"""
features.py — Feature analysis and clustering for BioLens.

After probing identifies concept directions, this module provides tools for:
- Dimensionality reduction (PCA, t-SNE, UMAP) of activation spaces
- Clustering of activations to discover emergent biological groupings
- Feature importance analysis (which neurons activate for which concepts)
- Superposition analysis (testing whether concepts share dimensions)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score


@dataclass
class DimensionalityReductionResult:
    """Result of projecting activations to low dimensions."""
    embeddings: np.ndarray       # [n_samples, n_components]
    method: str
    explained_variance: Optional[np.ndarray] = None  # PCA only
    labels: Optional[np.ndarray] = None

    @property
    def x(self) -> np.ndarray:
        return self.embeddings[:, 0]

    @property
    def y(self) -> np.ndarray:
        return self.embeddings[:, 1]


@dataclass
class SuperpositionAnalysis:
    """
    Results from superposition analysis.

    Superposition (Elhage et al., 2022) occurs when a model represents
    more concepts than it has dimensions, by encoding them as
    nearly-orthogonal directions that interfere slightly.

    In the biomedical domain: does the model pack drug mechanism,
    target family, and toxicity into overlapping directions?
    """
    n_concepts: int
    d_model: int
    pairwise_cosines: np.ndarray  # [n_concepts, n_concepts]
    mean_interference: float      # Average off-diagonal cosine similarity
    max_interference: float       # Maximum off-diagonal cosine similarity
    concept_names: list[str]
    compression_ratio: float      # n_concepts / d_model

    @property
    def is_superposed(self) -> bool:
        """Heuristic: superposition likely if interference > 0.1."""
        return self.mean_interference > 0.1


def reduce_dimensions(
    activations: np.ndarray,
    method: str = "pca",
    n_components: int = 2,
    labels: Optional[np.ndarray] = None,
    **kwargs,
) -> DimensionalityReductionResult:
    """
    Project high-dimensional activations to 2D/3D for visualization.

    Args:
        activations: [n_samples, d_model]
        method: "pca", "tsne", or "umap"
        n_components: Target dimensionality (2 or 3)
        labels: Optional labels for coloring
    """
    if method == "pca":
        reducer = PCA(n_components=n_components, random_state=42)
        embeddings = reducer.fit_transform(activations)
        return DimensionalityReductionResult(
            embeddings=embeddings,
            method="pca",
            explained_variance=reducer.explained_variance_ratio_,
            labels=labels,
        )

    elif method == "tsne":
        perplexity = kwargs.get("perplexity", min(30, len(activations) - 1))
        reducer = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=42,
            n_iter=1000,
        )
        embeddings = reducer.fit_transform(activations)
        return DimensionalityReductionResult(
            embeddings=embeddings,
            method="tsne",
            labels=labels,
        )

    elif method == "umap":
        try:
            import umap
            reducer = umap.UMAP(
                n_components=n_components,
                random_state=42,
                **kwargs,
            )
            embeddings = reducer.fit_transform(activations)
            return DimensionalityReductionResult(
                embeddings=embeddings,
                method="umap",
                labels=labels,
            )
        except ImportError:
            print("UMAP not installed. Install with: pip install umap-learn")
            print("Falling back to t-SNE.")
            return reduce_dimensions(activations, "tsne", n_components, labels)

    else:
        raise ValueError(f"Unknown method: {method}")


def analyze_superposition(
    concept_directions: dict[str, np.ndarray],
    d_model: int,
) -> SuperpositionAnalysis:
    """
    Analyze whether biological concepts are represented in superposition.

    Computes pairwise cosine similarities between concept directions.
    If concepts are orthogonal (cos ≈ 0), they're cleanly separated.
    If they have high cosine similarity, they're superposed.

    This is directly inspired by Elhage et al. (2022) "Toy Models of
    Superposition" — but applied to real biological concepts rather
    than synthetic features.
    """
    names = list(concept_directions.keys())
    n = len(names)
    directions = np.array([concept_directions[name] for name in names])

    # Normalize all directions
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    directions_normed = directions / norms

    # Pairwise cosine similarity matrix
    cosine_matrix = directions_normed @ directions_normed.T

    # Off-diagonal statistics
    mask = ~np.eye(n, dtype=bool)
    off_diagonal = np.abs(cosine_matrix[mask])

    return SuperpositionAnalysis(
        n_concepts=n,
        d_model=d_model,
        pairwise_cosines=cosine_matrix,
        mean_interference=float(np.mean(off_diagonal)),
        max_interference=float(np.max(off_diagonal)),
        concept_names=names,
        compression_ratio=n / d_model,
    )


def find_neuron_concept_associations(
    activations: np.ndarray,
    labels: np.ndarray,
    top_k: int = 20,
) -> dict[int, dict]:
    """
    Find individual neurons most associated with each class label.

    For each neuron, compute the difference in mean activation between
    classes. Neurons with large differences are "concept neurons" —
    they selectively activate for specific biological concepts.

    Returns:
        Dict mapping neuron index to {class_means, selectivity_score}.
    """
    unique_labels = np.unique(labels)
    n_neurons = activations.shape[1]

    # Compute mean activation per neuron per class
    class_means = np.zeros((len(unique_labels), n_neurons))
    for i, label in enumerate(unique_labels):
        class_means[i] = activations[labels == label].mean(axis=0)

    # Selectivity: max class mean minus mean of other class means
    selectivity = np.zeros(n_neurons)
    for neuron in range(n_neurons):
        means = class_means[:, neuron]
        selectivity[neuron] = np.max(means) - np.mean(means)

    # Top-k most selective neurons
    top_neurons = np.argsort(selectivity)[::-1][:top_k]

    results = {}
    for neuron_idx in top_neurons:
        preferred_class = unique_labels[np.argmax(class_means[:, neuron_idx])]
        results[int(neuron_idx)] = {
            "selectivity": float(selectivity[neuron_idx]),
            "preferred_class": int(preferred_class),
            "class_means": {
                int(label): float(class_means[i, neuron_idx])
                for i, label in enumerate(unique_labels)
            },
        }

    return results


def cluster_activations(
    activations: np.ndarray,
    method: str = "kmeans",
    n_clusters: Optional[int] = None,
    true_labels: Optional[np.ndarray] = None,
    **kwargs,
) -> dict:
    """
    Cluster activations to discover emergent groupings.

    If the model learns meaningful biological categories, unsupervised
    clustering should recover them. Comparing cluster assignments to
    true labels (via Adjusted Rand Index) measures this.
    """
    if method == "kmeans":
        if n_clusters is None:
            # Find optimal k via silhouette score
            best_k, best_score = 2, -1
            for k in range(2, min(10, len(activations))):
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                cluster_labels = km.fit_predict(activations)
                score = silhouette_score(activations, cluster_labels)
                if score > best_score:
                    best_k, best_score = k, score
            n_clusters = best_k

        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

    elif method == "dbscan":
        eps = kwargs.get("eps", 0.5)
        min_samples = kwargs.get("min_samples", 5)
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)

    else:
        raise ValueError(f"Unknown clustering method: {method}")

    cluster_labels = clusterer.fit_predict(activations)
    n_found = len(set(cluster_labels) - {-1})

    result = {
        "cluster_labels": cluster_labels,
        "n_clusters_found": n_found,
        "method": method,
    }

    if n_found > 1:
        valid_mask = cluster_labels >= 0
        if valid_mask.sum() > 1:
            result["silhouette"] = float(
                silhouette_score(activations[valid_mask], cluster_labels[valid_mask])
            )

    if true_labels is not None:
        result["adjusted_rand_index"] = float(
            adjusted_rand_score(true_labels, cluster_labels)
        )

    return result
