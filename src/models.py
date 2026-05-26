"""
models.py — Model loading and hooking utilities for BioLens.

Provides a unified interface for loading biomedical transformers
and attaching hooks for activation extraction.
"""

import torch
import torch.nn as nn
from typing import Optional, Callable
from dataclasses import dataclass, field
from transformers import AutoModel, AutoTokenizer

try:
    import transformer_lens as tl
    HAS_TRANSFORMER_LENS = True
except ImportError:
    HAS_TRANSFORMER_LENS = False


@dataclass
class ModelConfig:
    """Configuration for model loading."""
    model_name: str = "microsoft/biogpt"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float32
    cache_dir: Optional[str] = None
    use_transformer_lens: bool = True


@dataclass
class ActivationCache:
    """Stores activations captured during a forward pass."""
    residual_stream: dict[str, torch.Tensor] = field(default_factory=dict)
    attention_patterns: dict[str, torch.Tensor] = field(default_factory=dict)
    mlp_outputs: dict[str, torch.Tensor] = field(default_factory=dict)

    def clear(self):
        self.residual_stream.clear()
        self.attention_patterns.clear()
        self.mlp_outputs.clear()

    def to_device(self, device: str) -> "ActivationCache":
        """Move all cached tensors to a device."""
        for store in [self.residual_stream, self.attention_patterns, self.mlp_outputs]:
            for key in store:
                store[key] = store[key].to(device)
        return self


class BioModel:
    """
    Wrapper around a biomedical transformer for interpretability work.

    Supports two modes:
    1. TransformerLens mode (preferred): Full mechanistic interpretability toolkit
    2. HuggingFace mode (fallback): Manual hook-based activation extraction

    Example usage:
        >>> model = BioModel(ModelConfig(model_name="microsoft/biogpt"))
        >>> activations = model.extract_activations("Imatinib inhibits BCR-ABL")
        >>> print(activations.residual_stream.keys())
        dict_keys(['layer_0', 'layer_1', ..., 'layer_23'])
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.cache = ActivationCache()
        self._hooks: list = []

        if config.use_transformer_lens and HAS_TRANSFORMER_LENS:
            self._load_transformer_lens()
        else:
            self._load_huggingface()

    def _load_transformer_lens(self):
        """Load model via TransformerLens for full interpretability access."""
        print(f"Loading {self.config.model_name} via TransformerLens...")
        self.model = tl.HookedTransformer.from_pretrained(
            self.config.model_name,
            device=self.config.device,
            dtype=self.config.dtype,
            cache_dir=self.config.cache_dir,
        )
        self.tokenizer = self.model.tokenizer
        self.n_layers = self.model.cfg.n_layers
        self.d_model = self.model.cfg.d_model
        self.mode = "transformer_lens"
        print(f"Loaded: {self.n_layers} layers, d_model={self.d_model}")

    def _load_huggingface(self):
        """Fallback: load via HuggingFace with manual hooks."""
        print(f"Loading {self.config.model_name} via HuggingFace...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.cache_dir,
        )
        self.model = AutoModel.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.cache_dir,
            torch_dtype=self.config.dtype,
        ).to(self.config.device)
        self.model.eval()

        # Infer architecture details
        self.n_layers = self.model.config.num_hidden_layers
        self.d_model = self.model.config.hidden_size
        self.mode = "huggingface"
        print(f"Loaded: {self.n_layers} layers, d_model={self.d_model}")

    def extract_activations(
        self,
        text: str,
        layers: Optional[list[int]] = None,
        include_attention: bool = True,
        include_mlp: bool = True,
    ) -> ActivationCache:
        """
        Run a forward pass and extract intermediate activations.

        Args:
            text: Input text to process
            layers: Specific layers to extract (None = all layers)
            include_attention: Whether to capture attention patterns
            include_mlp: Whether to capture MLP outputs

        Returns:
            ActivationCache with extracted activations
        """
        self.cache.clear()

        if self.mode == "transformer_lens":
            return self._extract_tl(text, layers, include_attention, include_mlp)
        else:
            return self._extract_hf(text, layers, include_attention, include_mlp)

    def _extract_tl(self, text, layers, include_attention, include_mlp):
        """Extract activations using TransformerLens built-in caching."""
        names_filter = []
        target_layers = layers or list(range(self.n_layers))

        for layer in target_layers:
            names_filter.append(f"blocks.{layer}.hook_resid_post")
            if include_attention:
                names_filter.append(f"blocks.{layer}.attn.hook_pattern")
            if include_mlp:
                names_filter.append(f"blocks.{layer}.hook_mlp_out")

        _, tl_cache = self.model.run_with_cache(
            text,
            names_filter=names_filter,
        )

        for layer in target_layers:
            key = f"layer_{layer}"
            self.cache.residual_stream[key] = tl_cache[f"blocks.{layer}.hook_resid_post"]
            if include_attention:
                self.cache.attention_patterns[key] = tl_cache[f"blocks.{layer}.attn.hook_pattern"]
            if include_mlp:
                self.cache.mlp_outputs[key] = tl_cache[f"blocks.{layer}.hook_mlp_out"]

        return self.cache

    def _extract_hf(self, text, layers, include_attention, include_mlp):
        """Extract activations using HuggingFace hooks (fallback)."""
        tokens = self.tokenizer(text, return_tensors="pt").to(self.config.device)
        target_layers = layers or list(range(self.n_layers))

        hooks = []

        def make_hook(layer_idx: int, store_key: str, target_dict: dict):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    target_dict[f"layer_{layer_idx}"] = output[0].detach().cpu()
                else:
                    target_dict[f"layer_{layer_idx}"] = output.detach().cpu()
            return hook_fn

        # Register hooks on encoder/decoder layers
        encoder_layers = self._get_encoder_layers()
        for idx in target_layers:
            if idx < len(encoder_layers):
                layer = encoder_layers[idx]
                hooks.append(
                    layer.register_forward_hook(
                        make_hook(idx, f"layer_{idx}", self.cache.residual_stream)
                    )
                )

        with torch.no_grad():
            self.model(**tokens, output_attentions=include_attention)

        # Clean up hooks
        for h in hooks:
            h.remove()

        return self.cache

    def _get_encoder_layers(self) -> nn.ModuleList:
        """Find the transformer layers in the HuggingFace model."""
        # Common attribute names across architectures
        for attr in ["encoder.layer", "decoder.layers", "transformer.h", "model.layers", "layers"]:
            parts = attr.split(".")
            obj = self.model
            try:
                for part in parts:
                    obj = getattr(obj, part)
                return obj
            except AttributeError:
                continue
        raise ValueError(f"Could not find transformer layers in {type(self.model)}")

    def tokenize(self, text: str) -> list[str]:
        """Return human-readable tokens for a given text."""
        token_ids = self.tokenizer.encode(text)
        return [self.tokenizer.decode([tid]) for tid in token_ids]

    @property
    def device(self) -> str:
        return self.config.device

    def __repr__(self) -> str:
        return (
            f"BioModel(name={self.config.model_name}, "
            f"mode={self.mode}, "
            f"layers={self.n_layers}, "
            f"d_model={self.d_model})"
        )
