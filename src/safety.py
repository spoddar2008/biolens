"""
safety.py — Safety-relevant direction identification in activation space.

Core idea: if we can find a direction in activation space that separates
"benign biological reasoning" from "safety-sensitive biological reasoning,"
we have a potential internal monitor that's harder to circumvent than
behavioral (output-only) evaluation.

This is analogous to finding a biomarker that predicts treatment response
before clinical outcomes manifest — an early, internal signal.

Approach:
1. Collect activations for benign and sensitive prompts
2. Find the direction that maximally separates them (contrastive analysis)
3. Validate on held-out data
4. Analyze what biological concepts correlate with the safety direction
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc


@dataclass
class SafetyDirection:
    """A learned direction in activation space associated with safety-relevant content."""
    direction: np.ndarray       # Unit vector in activation space [d_model]
    layer: str                  # Which layer this was learned from
    auroc: float                # Area under ROC curve on held-out data
    auprc: float                # Area under precision-recall curve
    threshold: float            # Decision threshold for binary classification
    mean_benign_proj: float     # Mean projection of benign prompts onto direction
    mean_sensitive_proj: float  # Mean projection of sensitive prompts onto direction
    separation: float           # Cohen's d between benign and sensitive projections

    def score(self, activation: np.ndarray) -> float:
        """
        Score a single activation vector along the safety direction.

        Returns a scalar: higher values = more safety-relevant.
        """
        return float(np.dot(activation, self.direction))

    def classify(self, activation: np.ndarray) -> tuple[str, float]:
        """Classify an activation as benign or sensitive."""
        score = self.score(activation)
        label = "sensitive" if score > self.threshold else "benign"
        return label, score


@dataclass
class ContrastiveAnalysisResult:
    """Full results from contrastive safety analysis across layers."""
    directions: dict[str, SafetyDirection]  # layer_name -> direction
    best_layer: str
    best_auroc: float
    layer_aurocs: dict[str, float]

    def get_best_direction(self) -> SafetyDirection:
        return self.directions[self.best_layer]


class SafetyDirectionFinder:
    """
    Find directions in activation space that separate benign from
    safety-sensitive biological content.

    This uses contrastive activation analysis:
    1. Compute mean activation for benign prompts
    2. Compute mean activation for sensitive prompts
    3. The difference vector is the initial safety direction
    4. Refine with logistic regression for a maximum-margin direction
    5. Validate with cross-validation

    Example:
        >>> finder = SafetyDirectionFinder()
        >>> result = finder.analyze(
        ...     activations_by_layer=dataset.activations,
        ...     labels=dataset.labels["safety_level"],
        ...     benign_label=0,
        ...     sensitive_label=1,
        ... )
        >>> print(f"Best layer: {result.best_layer}, AUROC: {result.best_auroc:.3f}")
        >>> direction = result.get_best_direction()
    """

    def __init__(self, n_folds: int = 5, method: str = "logistic"):
        """
        Args:
            n_folds: Cross-validation folds for validation.
            method: "mean_diff" for simple difference vector,
                    "logistic" for maximum-margin direction (recommended).
        """
        self.n_folds = n_folds
        self.method = method

    def analyze(
        self,
        activations_by_layer: dict[str, np.ndarray],
        labels: np.ndarray,
        benign_label: int = 0,
        sensitive_label: int = 1,
    ) -> ContrastiveAnalysisResult:
        """
        Find safety directions across all layers.

        Args:
            activations_by_layer: Dict mapping layer names to [n_samples, d_model].
            labels: Binary labels (0=benign, 1=sensitive).
            benign_label: Integer label for benign class.
            sensitive_label: Integer label for sensitive class.

        Returns:
            ContrastiveAnalysisResult with directions and evaluation metrics.
        """
        # Filter to binary classification
        binary_mask = np.isin(labels, [benign_label, sensitive_label])
        binary_labels = (labels[binary_mask] == sensitive_label).astype(int)

        directions = {}
        layer_aurocs = {}

        print("Finding safety directions across layers...")
        for layer_name in sorted(activations_by_layer.keys(), key=_layer_key):
            acts = activations_by_layer[layer_name][binary_mask]

            direction_result = self._find_direction(
                acts, binary_labels, layer_name
            )
            directions[layer_name] = direction_result
            layer_aurocs[layer_name] = direction_result.auroc

            print(f"  {layer_name}: AUROC={direction_result.auroc:.3f}, "
                  f"separation={direction_result.separation:.2f}")

        best_layer = max(layer_aurocs, key=layer_aurocs.get)

        return ContrastiveAnalysisResult(
            directions=directions,
            best_layer=best_layer,
            best_auroc=layer_aurocs[best_layer],
            layer_aurocs=layer_aurocs,
        )

    def _find_direction(
        self,
        activations: np.ndarray,
        labels: np.ndarray,
        layer_name: str,
    ) -> SafetyDirection:
        """Find the safety direction for a single layer."""

        if self.method == "mean_diff":
            return self._mean_diff_direction(activations, labels, layer_name)
        elif self.method == "logistic":
            return self._logistic_direction(activations, labels, layer_name)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _mean_diff_direction(
        self, acts: np.ndarray, labels: np.ndarray, layer: str
    ) -> SafetyDirection:
        """Simple mean-difference direction."""
        benign_mean = acts[labels == 0].mean(axis=0)
        sensitive_mean = acts[labels == 1].mean(axis=0)

        direction = sensitive_mean - benign_mean
        direction = direction / np.linalg.norm(direction)

        return self._evaluate_direction(direction, acts, labels, layer)

    def _logistic_direction(
        self, acts: np.ndarray, labels: np.ndarray, layer: str
    ) -> SafetyDirection:
        """Maximum-margin direction via logistic regression."""
        scaler = StandardScaler()
        acts_scaled = scaler.fit_transform(acts)

        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            C=1.0,
            random_state=42,
        )

        # Cross-validate — guard against single-class or too-small datasets
        n_classes = len(np.unique(labels))
        if n_classes < 2:
            # All same class — skip CV, return dummy direction
            direction = np.zeros(acts.shape[1])
            result = self._evaluate_direction(direction, acts, labels, layer)
            result.auroc = 0.5
            return result

        n_splits = max(2, min(self.n_folds, int(np.bincount(labels).min())))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = cross_val_score(clf, acts_scaled, labels, cv=cv, scoring="roc_auc")

        # Fit on all data for direction extraction
        clf.fit(acts_scaled, labels)

        # Unscale direction back to original space
        direction = (clf.coef_[0] / scaler.scale_).astype(np.float64)
        direction = direction / np.linalg.norm(direction)

        result = self._evaluate_direction(direction, acts, labels, layer)
        # Use CV AUROC instead of in-sample
        result.auroc = cv_scores.mean()
        return result

    def _evaluate_direction(
        self,
        direction: np.ndarray,
        acts: np.ndarray,
        labels: np.ndarray,
        layer: str,
    ) -> SafetyDirection:
        """Evaluate a direction's ability to separate benign from sensitive."""
        # Project all activations onto the direction
        projections = acts @ direction

        benign_projs = projections[labels == 0]
        sensitive_projs = projections[labels == 1]

        # Cohen's d for effect size
        pooled_std = np.sqrt(
            (np.var(benign_projs) + np.var(sensitive_projs)) / 2
        )
        separation = (
            (np.mean(sensitive_projs) - np.mean(benign_projs)) / pooled_std
            if pooled_std > 0 else 0.0
        )

        # AUROC
        try:
            auroc = roc_auc_score(labels, projections)
        except ValueError:
            auroc = 0.5

        # AUPRC
        try:
            precision, recall, _ = precision_recall_curve(labels, projections)
            auprc = auc(recall, precision)
        except ValueError:
            auprc = 0.5

        # Threshold (Youden's J)
        threshold = (np.mean(benign_projs) + np.mean(sensitive_projs)) / 2

        return SafetyDirection(
            direction=direction,
            layer=layer,
            auroc=auroc,
            auprc=auprc,
            threshold=threshold,
            mean_benign_proj=float(np.mean(benign_projs)),
            mean_sensitive_proj=float(np.mean(sensitive_projs)),
            separation=float(separation),
        )


def correlate_with_concepts(
    safety_direction: SafetyDirection,
    concept_directions: dict[str, np.ndarray],
) -> dict[str, float]:
    """
    Measure how the safety direction correlates with other biological concepts.

    This answers: "Does the model's 'dangerous knowledge' direction overlap
    with its 'toxicity' direction or its 'synthesis pathway' direction?"

    Args:
        safety_direction: The learned safety direction.
        concept_directions: Dict mapping concept names to their directions
                           (from linear probing).

    Returns:
        Dict mapping concept names to cosine similarity with safety direction.
    """
    correlations = {}
    for concept_name, concept_dir in concept_directions.items():
        cos_sim = np.dot(safety_direction.direction, concept_dir) / (
            np.linalg.norm(safety_direction.direction) * np.linalg.norm(concept_dir)
        )
        correlations[concept_name] = float(cos_sim)

    return dict(sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True))


def _layer_key(name: str) -> int:
    try:
        return int(name.split("_")[1])
    except (IndexError, ValueError):
        return 0
