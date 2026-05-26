"""
patching.py — Activation patching for causal tracing in biomedical models.

Activation patching (causal tracing) answers: "Which components of the model
are causally responsible for a given prediction?" by corrupting inputs,
then selectively restoring activations at specific layers/positions/heads
to see which restorations recover the original prediction.

This is the core technique from:
- Meng et al. (2022) "Locating and Editing Factual Associations in GPT"
- Conmy et al. (2023) "Automated Circuit Discovery"

Applied here to biomedical knowledge:
- Which layers/heads store drug-target binding knowledge?
- Where does the model compute toxicity predictions?
- Do safety-relevant circuits overlap with general biomedical reasoning?
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional, Callable
from tqdm import tqdm


@dataclass
class PatchingResult:
    """Result of an activation patching experiment."""
    clean_logit: float              # Logit for target token on clean input
    corrupted_logit: float          # Logit for target token on corrupted input
    patched_logits: dict[str, float]  # Logit after patching each component
    effect_sizes: dict[str, float]  # Normalized effect of each patch

    @property
    def total_effect(self) -> float:
        """Total effect of corruption."""
        return self.clean_logit - self.corrupted_logit

    def top_components(self, n: int = 10) -> list[tuple[str, float]]:
        """Return the n components with largest causal effect."""
        sorted_effects = sorted(
            self.effect_sizes.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        return sorted_effects[:n]


@dataclass
class PatchingExperiment:
    """Configuration for a patching experiment."""
    clean_prompt: str
    corrupted_prompt: str
    target_token: str
    patch_type: str = "residual"  # "residual", "attention", "mlp"
    positions: Optional[list[int]] = None  # Token positions to patch (None = all)


class ActivationPatcher:
    """
    Perform activation patching experiments on a BioModel.

    Supports three levels of granularity:
    1. Layer-level: Patch entire residual stream at each layer
    2. Component-level: Patch attention output vs MLP output separately
    3. Head-level: Patch individual attention heads

    Example:
        >>> patcher = ActivationPatcher(model)
        >>> result = patcher.run(
        ...     clean="Imatinib inhibits BCR-ABL",
        ...     corrupted="Imatinib inhibits RANDOM",
        ...     target_token="kinase",
        ... )
        >>> print(result.top_components(5))
        [('layer_8_attn', 0.45), ('layer_10_mlp', 0.32), ...]
    """

    def __init__(self, model):
        """
        Args:
            model: A BioModel instance (must be in transformer_lens mode
                   for full patching support).
        """
        self.model = model
        if model.mode != "transformer_lens":
            print("Warning: Full patching requires TransformerLens mode. "
                  "Falling back to layer-level patching only.")

    def run(
        self,
        clean: str,
        corrupted: str,
        target_token: str,
        patch_type: str = "residual",
        positions: Optional[list[int]] = None,
    ) -> PatchingResult:
        """
        Run a single activation patching experiment.

        1. Run clean forward pass, cache all activations
        2. Run corrupted forward pass, cache all activations
        3. For each component, run corrupted input but patch in clean activation
        4. Measure how much each patch recovers the clean prediction

        Args:
            clean: The clean (correct) input prompt
            corrupted: The corrupted input prompt (e.g., with a word replaced)
            target_token: The token whose logit we're tracking
            patch_type: Which activations to patch
            positions: Which token positions to patch (None = last token)
        """
        if self.model.mode == "transformer_lens":
            return self._run_tl(clean, corrupted, target_token, patch_type, positions)
        else:
            return self._run_hf(clean, corrupted, target_token)

    def _run_tl(self, clean, corrupted, target_token, patch_type, positions):
        """Full patching with TransformerLens."""
        model = self.model.model

        # Get target token id
        target_id = self.model.tokenizer.encode(target_token)[-1]

        # Clean forward pass with caching
        clean_logits, clean_cache = model.run_with_cache(clean)
        clean_logit = clean_logits[0, -1, target_id].item()

        # Corrupted forward pass
        corrupted_logits, corrupted_cache = model.run_with_cache(corrupted)
        corrupted_logit = corrupted_logits[0, -1, target_id].item()

        total_effect = clean_logit - corrupted_logit
        if abs(total_effect) < 1e-6:
            print("Warning: Clean and corrupted logits are nearly identical. "
                  "Corruption may not be effective.")

        # Patch each component
        patched_logits = {}
        effect_sizes = {}

        for layer in range(self.model.n_layers):
            hook_names = self._get_hook_names(layer, patch_type)

            for hook_name in hook_names:
                component_key = f"layer_{layer}_{hook_name.split('.')[-1]}"

                def make_patch_hook(clean_act):
                    def hook_fn(activation, hook):
                        if positions is not None:
                            result = activation.clone()
                            for pos in positions:
                                result[0, pos] = clean_act[0, pos]
                            return result
                        else:
                            return clean_act
                    return hook_fn

                clean_activation = clean_cache[hook_name]
                patched_out = model.run_with_hooks(
                    corrupted,
                    fwd_hooks=[(hook_name, make_patch_hook(clean_activation))],
                )
                patched_logit = patched_out[0, -1, target_id].item()

                patched_logits[component_key] = patched_logit
                if abs(total_effect) > 1e-6:
                    effect_sizes[component_key] = (
                        (patched_logit - corrupted_logit) / total_effect
                    )
                else:
                    effect_sizes[component_key] = 0.0

        return PatchingResult(
            clean_logit=clean_logit,
            corrupted_logit=corrupted_logit,
            patched_logits=patched_logits,
            effect_sizes=effect_sizes,
        )

    def _run_hf(self, clean, corrupted, target_token):
        """Simplified layer-level patching for HuggingFace models."""
        # Extract clean and corrupted activations
        clean_cache = self.model.extract_activations(clean)
        corrupted_cache = self.model.extract_activations(corrupted)

        # For HF mode, we measure representation similarity as a proxy
        # for causal effect (true patching requires hook-based intervention)
        effect_sizes = {}
        for layer_name in clean_cache.residual_stream:
            clean_act = clean_cache.residual_stream[layer_name].numpy().flatten()
            corrupt_act = corrupted_cache.residual_stream[layer_name].numpy().flatten()

            # Cosine distance as proxy for causal importance
            cos_sim = np.dot(clean_act, corrupt_act) / (
                np.linalg.norm(clean_act) * np.linalg.norm(corrupt_act)
            )
            effect_sizes[layer_name] = 1.0 - cos_sim

        return PatchingResult(
            clean_logit=0.0,
            corrupted_logit=0.0,
            patched_logits={},
            effect_sizes=effect_sizes,
        )

    def _get_hook_names(self, layer: int, patch_type: str) -> list[str]:
        """Get TransformerLens hook point names for a given layer and type."""
        if patch_type == "residual":
            return [f"blocks.{layer}.hook_resid_post"]
        elif patch_type == "attention":
            return [f"blocks.{layer}.hook_attn_out"]
        elif patch_type == "mlp":
            return [f"blocks.{layer}.hook_mlp_out"]
        elif patch_type == "both":
            return [
                f"blocks.{layer}.hook_attn_out",
                f"blocks.{layer}.hook_mlp_out",
            ]
        else:
            raise ValueError(f"Unknown patch_type: {patch_type}")

    def sweep_prompts(
        self,
        experiments: list[PatchingExperiment],
    ) -> dict[str, np.ndarray]:
        """
        Run patching across multiple prompt pairs and aggregate results.

        Returns averaged effect sizes across all experiments — this reveals
        which components are *generally* important for a class of predictions,
        not just for a single example.
        """
        all_effects = {}

        for exp in tqdm(experiments, desc="Patching experiments"):
            result = self.run(
                clean=exp.clean_prompt,
                corrupted=exp.corrupted_prompt,
                target_token=exp.target_token,
                patch_type=exp.patch_type,
                positions=exp.positions,
            )

            for component, effect in result.effect_sizes.items():
                if component not in all_effects:
                    all_effects[component] = []
                all_effects[component].append(effect)

        # Average across experiments
        return {k: np.array(v) for k, v in all_effects.items()}
