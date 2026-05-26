"""
probes.py — Linear probing classifiers for biological concept detection.

Core idea: if a linear classifier can decode a concept (e.g., "this drug is a 
kinase inhibitor") from a model's intermediate activations, then that concept 
is represented as a direction in activation space.

This is the standard approach from mechanistic interpretability 
(Alain & Bengio, 2017) applied to biomedical knowledge.
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix


@dataclass
class ProbeResult:
    """Results from training a linear probe."""
    concept: str
    layer: str
    accuracy: float
    cv_scores: np.ndarray
    cv_mean: float
    cv_std: float
    classifier: Pipeline
    report: str
    direction: Optional[np.ndarray] = None  # The learned concept direction

    def __repr__(self) -> str:
        return (
            f"ProbeResult(concept='{self.concept}', layer='{self.layer}', "
            f"cv_accuracy={self.cv_mean:.3f} ± {self.cv_std:.3f})"
        )


class BiologicalProbe:
    """
    Train linear probes to detect biological concepts in model activations.

    Supported concept types (extend as needed):
    - drug_mechanism: kinase inhibitor, antimetabolite, immunotherapy, etc.
    - target_family: receptor tyrosine kinase, nuclear receptor, ion channel, etc.
    - toxicity_class: hepatotoxic, cardiotoxic, neurotoxic, nephrotoxic
    - pathway: glycolysis, TCA cycle, nucleotide biosynthesis, etc.
    - safety_relevant: benign query vs. dual-use relevant query

    Example:
        >>> probe = BiologicalProbe(concept="drug_mechanism")
        >>> result = probe.train(activations, labels, layer="layer_12")
        >>> print(result)
        ProbeResult(concept='drug_mechanism', layer='layer_12', cv_accuracy=0.847 ± 0.032)
        >>> direction = result.direction  # The concept direction in activation space
    """

    def __init__(
        self,
        concept: str,
        n_folds: int = 5,
        max_iter: int = 1000,
        C: float = 1.0,
    ):
        self.concept = concept
        self.n_folds = n_folds
        self.max_iter = max_iter
        self.C = C

    def train(
        self,
        activations: np.ndarray,
        labels: np.ndarray,
        layer: str,
    ) -> ProbeResult:
        """
        Train a linear probe on activations from a specific layer.

        Args:
            activations: Array of shape (n_samples, d_model) — the residual 
                         stream activations at a given layer for each input.
            labels: Array of shape (n_samples,) — integer class labels.
            layer: Layer identifier (e.g., "layer_12").

        Returns:
            ProbeResult with accuracy, cross-validation scores, and the 
            learned concept direction.
        """
        assert len(activations) == len(labels), "Activation/label count mismatch"
        assert activations.ndim == 2, f"Expected 2D activations, got {activations.ndim}D"

        # Build pipeline: standardize then classify
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=self.max_iter,
                C=self.C,
                class_weight="balanced",
                random_state=42,
            )),
        ])

        # Cross-validation — guard against single-class or too-small datasets
        unique, counts = np.unique(labels, return_counts=True)
        n_classes = len(unique)
        if n_classes < 2:
            # Only one class present — classifier is undefined, return dummy
            direction = np.zeros(activations.shape[1])
            return ProbeResult(
                concept=self.concept,
                layer=layer,
                accuracy=0.0,
                cv_scores=np.array([0.0]),
                cv_mean=0.0,
                cv_std=0.0,
                classifier=None,
                report="Skipped: only one class in labels",
                direction=direction,
            )

        min_class_count = int(counts.min())
        n_splits = min(self.n_folds, min_class_count)
        if n_splits < 2:
            cv_scores = np.array([0.0])
        else:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_scores = cross_val_score(pipeline, activations, labels, cv=cv, scoring="accuracy")

        # Fit on full data for direction extraction
        pipeline.fit(activations, labels)

        # Extract the concept direction (weight vector of the linear classifier)
        coef = pipeline.named_steps["classifier"].coef_
        scaler = pipeline.named_steps["scaler"]

        # Unscale the direction back to activation space
        # w_original = w_scaled / scale (since scaler transforms x -> (x - mean) / scale)
        direction = (coef / scaler.scale_[np.newaxis, :]).squeeze()

        # Normalize to unit vector
        direction = direction / np.linalg.norm(direction)

        # Classification report
        predictions = pipeline.predict(activations)
        report = classification_report(labels, predictions)

        return ProbeResult(
            concept=self.concept,
            layer=layer,
            accuracy=np.mean(predictions == labels),
            cv_scores=cv_scores,
            cv_mean=cv_scores.mean(),
            cv_std=cv_scores.std(),
            classifier=pipeline,
            report=report,
            direction=direction,
        )

    def sweep_layers(
        self,
        activations_by_layer: dict[str, np.ndarray],
        labels: np.ndarray,
    ) -> list[ProbeResult]:
        """
        Train probes across all layers to find where a concept is best represented.

        This is the key experiment: it reveals the layer at which the model 
        transitions from surface-level token patterns to meaningful biological 
        abstraction.

        Args:
            activations_by_layer: Dict mapping layer names to activation arrays.
            labels: Shared label array for all layers.

        Returns:
            List of ProbeResults, one per layer, sorted by layer order.
        """
        results = []
        for layer_name in sorted(activations_by_layer.keys(), key=_layer_sort_key):
            acts = activations_by_layer[layer_name]
            result = self.train(acts, labels, layer=layer_name)
            results.append(result)
            print(f"  {layer_name}: {result.cv_mean:.3f} ± {result.cv_std:.3f}")

        return results


def compare_directions(
    direction_a: np.ndarray,
    direction_b: np.ndarray,
) -> float:
    """
    Compute cosine similarity between two concept directions.

    Useful for asking: does the "toxicity" direction overlap with the 
    "dual-use knowledge" direction? High similarity suggests the model 
    entangles these concepts.
    """
    return float(np.dot(direction_a, direction_b) / (
        np.linalg.norm(direction_a) * np.linalg.norm(direction_b)
    ))


def _layer_sort_key(layer_name: str) -> int:
    """Extract numeric index from layer name for sorting."""
    try:
        return int(layer_name.split("_")[1])
    except (IndexError, ValueError):
        return 0
