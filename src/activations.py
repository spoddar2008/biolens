"""
activations.py — Activation extraction and caching for BioLens.

Handles batch extraction of activations across a dataset of prompts,
with disk caching to avoid re-running expensive forward passes.
Supports extracting residual stream, attention patterns, and MLP outputs
at configurable granularity.
"""

import json
import hashlib
import numpy as np
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from tqdm import tqdm


@dataclass
class ActivationDataset:
    """
    A dataset of activations extracted from a model for a set of prompts.

    Attributes:
        activations: Dict mapping layer names to arrays of shape
                     [n_prompts, d_model] (using mean-pooled residual stream).
        labels: Dict mapping concept names to integer label arrays.
        texts: List of original prompt texts.
        token_level: If True, activations are [n_prompts, max_seq_len, d_model].
        metadata: Dict of additional per-prompt metadata.
    """
    activations: dict[str, np.ndarray]
    labels: dict[str, np.ndarray]
    texts: list[str]
    token_level: bool = False
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @property
    def n_prompts(self) -> int:
        return len(self.texts)

    @property
    def layers(self) -> list[str]:
        return sorted(self.activations.keys(), key=_layer_sort_key)

    @property
    def d_model(self) -> int:
        first_key = next(iter(self.activations))
        return self.activations[first_key].shape[-1]

    def get_labeled_subset(
        self,
        concept: str,
        labels_to_keep: Optional[list[int]] = None,
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Get activations and labels for a specific concept, optionally filtered."""
        concept_labels = self.labels[concept]
        if labels_to_keep is not None:
            mask = np.isin(concept_labels, labels_to_keep)
            filtered_acts = {k: v[mask] for k, v in self.activations.items()}
            filtered_labels = concept_labels[mask]
            return filtered_acts, filtered_labels
        return self.activations, concept_labels

    def save(self, path: str):
        """Save dataset to disk as compressed npz + json."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save activations as compressed numpy
        np.savez_compressed(
            path / "activations.npz",
            **self.activations,
        )

        # Save labels
        np.savez_compressed(
            path / "labels.npz",
            **self.labels,
        )

        # Save metadata
        meta = {
            "texts": self.texts,
            "token_level": self.token_level,
            "metadata": self.metadata,
            "n_prompts": self.n_prompts,
            "d_model": self.d_model,
            "layers": self.layers,
        }
        with open(path / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"Saved ActivationDataset ({self.n_prompts} prompts, "
              f"{len(self.layers)} layers) to {path}")

    @classmethod
    def load(cls, path: str) -> "ActivationDataset":
        """Load dataset from disk."""
        path = Path(path)

        acts_data = np.load(path / "activations.npz")
        activations = {k: acts_data[k] for k in acts_data.files}

        labels_data = np.load(path / "labels.npz")
        labels = {k: labels_data[k] for k in labels_data.files}

        with open(path / "meta.json") as f:
            meta = json.load(f)

        return cls(
            activations=activations,
            labels=labels,
            texts=meta["texts"],
            token_level=meta.get("token_level", False),
            metadata=meta.get("metadata", {}),
        )


class ActivationExtractor:
    """
    Extract activations from a BioModel for a dataset of prompts.

    Supports two pooling strategies:
    - "mean": Mean-pool across token positions → [d_model] per prompt
    - "last": Use last token position → [d_model] per prompt
    - "token": Keep all positions → [seq_len, d_model] per prompt

    Example:
        >>> from src.models import BioModel, ModelConfig
        >>> model = BioModel(ModelConfig(model_name="microsoft/biogpt"))
        >>> extractor = ActivationExtractor(model, pooling="mean")
        >>> dataset = extractor.extract_from_jsonl("data/prompts/drug_target_pairs.jsonl")
    """

    def __init__(
        self,
        model,
        pooling: str = "mean",
        layers: Optional[list[int]] = None,
        batch_size: int = 8,
        cache_dir: Optional[str] = None,
    ):
        self.model = model
        self.pooling = pooling
        self.layers = layers
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def extract_from_jsonl(
        self,
        jsonl_path: str,
        text_field: str = "text",
        label_fields: Optional[list[str]] = None,
    ) -> ActivationDataset:
        """
        Extract activations for all prompts in a JSONL file.

        Args:
            jsonl_path: Path to JSONL file with prompts and labels.
            text_field: Key in each JSON object containing the text.
            label_fields: Keys to extract as classification labels.

        Returns:
            ActivationDataset with activations and labels.
        """
        # Check cache first
        cache_key = self._cache_key(jsonl_path)
        if self.cache_dir and (self.cache_dir / cache_key).exists():
            print(f"Loading cached activations from {self.cache_dir / cache_key}")
            return ActivationDataset.load(str(self.cache_dir / cache_key))

        # Load prompts
        records = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        texts = [r[text_field] for r in records]

        # Determine label fields
        if label_fields is None:
            label_fields = [k for k in records[0].keys() if k != text_field]

        # Build label encodings
        labels = {}
        label_maps = {}
        for field in label_fields:
            unique_values = sorted(set(r.get(field, "unknown") for r in records))
            value_to_int = {v: i for i, v in enumerate(unique_values)}
            labels[field] = np.array([value_to_int[r.get(field, "unknown")] for r in records])
            label_maps[field] = {v: i for v, i in value_to_int.items()}

        # Extract activations
        all_activations = {}  # layer_name -> list of arrays
        print(f"Extracting activations for {len(texts)} prompts...")

        for i, text in enumerate(tqdm(texts)):
            cache = self.model.extract_activations(
                text,
                layers=self.layers,
                include_attention=False,
                include_mlp=False,
            )

            for layer_name, tensor in cache.residual_stream.items():
                if layer_name not in all_activations:
                    all_activations[layer_name] = []

                pooled = self._pool(tensor)
                all_activations[layer_name].append(pooled)

        # Stack into arrays
        activations = {
            k: np.stack(v, axis=0) for k, v in all_activations.items()
        }

        dataset = ActivationDataset(
            activations=activations,
            labels=labels,
            texts=texts,
            token_level=(self.pooling == "token"),
            metadata={"label_maps": label_maps, "source": jsonl_path},
        )

        # Cache if configured
        if self.cache_dir:
            dataset.save(str(self.cache_dir / cache_key))

        return dataset

    def extract_from_texts(
        self,
        texts: list[str],
        labels: Optional[dict[str, np.ndarray]] = None,
    ) -> ActivationDataset:
        """Extract activations from a list of texts (no file needed)."""
        all_activations = {}

        for text in tqdm(texts, desc="Extracting"):
            cache = self.model.extract_activations(
                text, layers=self.layers,
                include_attention=False, include_mlp=False,
            )
            for layer_name, tensor in cache.residual_stream.items():
                if layer_name not in all_activations:
                    all_activations[layer_name] = []
                all_activations[layer_name].append(self._pool(tensor))

        activations = {k: np.stack(v) for k, v in all_activations.items()}

        return ActivationDataset(
            activations=activations,
            labels=labels or {},
            texts=texts,
        )

    def _pool(self, tensor: torch.Tensor) -> np.ndarray:
        """Pool a [1, seq_len, d_model] tensor to [d_model]."""
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu().numpy()

        # Remove batch dimension if present
        if tensor.ndim == 3:
            tensor = tensor[0]  # [seq_len, d_model]

        if self.pooling == "mean":
            return tensor.mean(axis=0)
        elif self.pooling == "last":
            return tensor[-1]
        elif self.pooling == "token":
            return tensor
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

    def _cache_key(self, jsonl_path: str) -> str:
        """Generate a cache key based on model + data + config."""
        key_str = f"{self.model.config.model_name}_{jsonl_path}_{self.pooling}"
        return hashlib.md5(key_str.encode()).hexdigest()[:12]


def _layer_sort_key(layer_name: str) -> int:
    try:
        return int(layer_name.split("_")[1])
    except (IndexError, ValueError):
        return 0
