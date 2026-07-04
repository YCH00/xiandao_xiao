"""Baseline-derived compressed Llama classifier for FinGPT labels.

This module is the compliant lightweight model path: the student is initialized
from the official Llama2 base model with the FinGPT LoRA adapter merged, then a
subset of Transformer layers is kept and a small 13-class classification head is
trained by distillation/fine-tuning.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaModel

ALL_LABELS: tuple[str, ...] = (
    "上涨0-1%",
    "上涨1-2%",
    "上涨2-3%",
    "上涨3-4%",
    "上涨4-5%",
    "上涨超过5%",
    "下跌0-1%",
    "下跌1-2%",
    "下跌2-3%",
    "下跌3-4%",
    "下跌4-5%",
    "下跌超过5%",
    "股价持平",
)

DIRECTION_LABELS: tuple[str, ...] = ("up", "down", "flat")


def label_direction(label: str | None) -> str | None:
    if label is None:
        return None
    if label.startswith("上涨"):
        return "up"
    if label.startswith("下跌"):
        return "down"
    return "flat"


def label_to_direction_id(label: str) -> int:
    direction = label_direction(label)
    return DIRECTION_LABELS.index(direction or "flat")


def uniform_layer_indices(total_layers: int, kept_layers: int) -> list[int]:
    if kept_layers <= 0:
        raise ValueError("kept_layers must be positive")
    if kept_layers > total_layers:
        raise ValueError("kept_layers cannot exceed total_layers")
    if kept_layers == total_layers:
        return list(range(total_layers))
    # Include early/mid/late representations instead of taking only shallow layers.
    return sorted({round(i * (total_layers - 1) / (kept_layers - 1)) for i in range(kept_layers)})


def parse_layer_indices(value: str | None, *, total_layers: int, kept_layers: int) -> list[int]:
    if value:
        indices = [int(x.strip()) for x in value.split(",") if x.strip()]
    else:
        indices = uniform_layer_indices(total_layers, kept_layers)
    if not indices:
        raise ValueError("No layer indices selected")
    if min(indices) < 0 or max(indices) >= total_layers:
        raise ValueError(f"Layer indices must be within [0, {total_layers - 1}]")
    if len(set(indices)) != len(indices):
        raise ValueError("Layer indices must be unique")
    return indices


class CompressedLlamaClassifier(nn.Module):
    def __init__(
        self,
        config: LlamaConfig,
        *,
        labels: Iterable[str] = ALL_LABELS,
        max_length: int = 4096,
    ) -> None:
        super().__init__()
        # The judging ROCm image may not provide flash_attn_2_cuda*.so.
        # Eager attention is slower but portable and matches the official fallback path.
        config._attn_implementation = "eager"
        self.labels = tuple(labels)
        self.max_length = int(max_length)
        self.model = LlamaModel(config)
        self.classifier = nn.Linear(config.hidden_size, len(self.labels))
        self.direction_classifier = nn.Linear(config.hidden_size, len(DIRECTION_LABELS))

    def _pool_last_token(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        if attention_mask is None:
            return hidden_states[:, -1]
        lengths = attention_mask.long().sum(dim=1).clamp(min=1) - 1
        batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_idx, lengths]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        pooled = self._pool_last_token(outputs.last_hidden_state, attention_mask)
        return {
            "logits": self.classifier(pooled),
            "direction_logits": self.direction_classifier(pooled),
        }

    def predict_label(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> str:
        outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        index = int(outputs["logits"].argmax(dim=-1).item())
        return self.labels[index]

    def save_classifier(
        self,
        output_dir: str | Path,
        *,
        layer_indices: list[int],
        base_model_path: str,
        adapter_path: str,
    ) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), output_dir / "pytorch_model.bin")
        metadata = {
            "model_type": "compressed_llama_classifier",
            "labels": list(self.labels),
            "direction_labels": list(DIRECTION_LABELS),
            "max_length": self.max_length,
            "layer_indices": layer_indices,
            "base_model_path": base_model_path,
            "adapter_path": adapter_path,
            "config": self.model.config.to_dict(),
        }
        (output_dir / "classifier_config.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load_classifier(
        cls,
        model_dir: str | Path,
        *,
        map_location: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
    ) -> "CompressedLlamaClassifier":
        model_dir = Path(model_dir)
        metadata = json.loads((model_dir / "classifier_config.json").read_text(encoding="utf-8"))
        config = LlamaConfig(**metadata["config"])
        model = cls(
            config,
            labels=tuple(metadata["labels"]),
            max_length=int(metadata.get("max_length", 4096)),
        )
        weights_path = model_dir / "pytorch_model.bin"
        try:
            state = torch.load(weights_path, map_location=map_location, weights_only=True)
        except TypeError:
            state = torch.load(weights_path, map_location=map_location)
        model.load_state_dict(state)
        if dtype is not None:
            model.to(dtype=dtype)
        return model


def create_compressed_classifier_from_causal_lm(
    source_model,
    *,
    layer_indices: list[int],
    max_length: int,
) -> CompressedLlamaClassifier:
    """Create a compressed classifier initialized from a LlamaForCausalLM model."""
    source_core = source_model.model
    config = copy.deepcopy(source_model.config)
    config.num_hidden_layers = len(layer_indices)
    config.use_cache = False

    student = CompressedLlamaClassifier(config, labels=ALL_LABELS, max_length=max_length)
    dtype = next(source_core.parameters()).dtype
    student.to(dtype=dtype)

    student.model.embed_tokens.load_state_dict(source_core.embed_tokens.state_dict())
    student.model.norm.load_state_dict(source_core.norm.state_dict())
    for dst_idx, src_idx in enumerate(layer_indices):
        student.model.layers[dst_idx].load_state_dict(source_core.layers[src_idx].state_dict())
    return student


def set_trainable_layers(model: CompressedLlamaClassifier, *, unfreeze_last_layers: int) -> None:
    for param in model.model.parameters():
        param.requires_grad = False
    if unfreeze_last_layers > 0:
        for layer in model.model.layers[-unfreeze_last_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
        for param in model.model.norm.parameters():
            param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True
    for param in model.direction_classifier.parameters():
        param.requires_grad = True
