#!/usr/bin/env python3
"""Train a baseline-derived compressed Llama classifier.

This is the compliant second-layer route:
  official Llama2-7B + FinGPT LoRA -> merge LoRA -> keep selected Llama layers
  -> train 13-class and direction heads with labels/teacher hard distillation.
"""
from __future__ import annotations

import argparse
import gc
import glob
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from compressed_llama_classifier import (
    ALL_LABELS,
    DIRECTION_LABELS,
    CompressedLlamaClassifier,
    create_compressed_classifier_from_causal_lm,
    label_direction,
    label_to_direction_id,
    parse_layer_indices,
    set_trainable_layers,
)
from scoring.metrics import direction_acc, macro_f1


LABEL_TO_ID = {label: i for i, label in enumerate(ALL_LABELS)}


def _expand_paths(patterns: list[str]) -> list[str]:
    paths: list[str] = []
    for pattern in patterns:
        paths.extend(sorted(glob.glob(pattern)))
    if not paths:
        raise SystemExit(f"No parquet files matched: {patterns}")
    return list(dict.fromkeys(paths))


def _load_rows(patterns: list[str], limit: int) -> tuple[list[str], list[str]]:
    import pandas as pd

    frames = [pd.read_parquet(path, columns=["prompt", "label"]) for path in _expand_paths(patterns)]
    df = pd.concat(frames, ignore_index=True).dropna(subset=["prompt", "label"])
    df = df[df["label"].isin(ALL_LABELS)]
    if limit > 0:
        df = df.head(limit)
    if df.empty:
        raise SystemExit("No valid prompt/label rows found")
    return df["prompt"].astype(str).tolist(), df["label"].astype(str).tolist()


def _load_teacher_labels(path: str | None, n: int) -> list[str | None]:
    teacher_labels: list[str | None] = [None] * n
    if not path:
        return teacher_labels
    teacher_path = Path(path)
    if not teacher_path.exists():
        print(f"teacher label file not found: {teacher_path}; training without teacher labels", flush=True)
        return teacher_labels
    with teacher_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            record = json.loads(line)
            label = record.get("teacher_label")
            if label in LABEL_TO_ID:
                teacher_labels[i] = label
    return teacher_labels


def _stratified_split(labels: list[str], *, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels_arr = np.array(labels)
    train_indices: list[int] = []
    val_indices: list[int] = []
    for label in ALL_LABELS:
        idx = np.where(labels_arr == label)[0]
        rng.shuffle(idx)
        if val_ratio <= 0 or len(idx) <= 1:
            train_indices.extend(idx.tolist())
            continue
        n_val = max(1, int(round(len(idx) * val_ratio)))
        n_val = min(n_val, len(idx) - 1)
        val_indices.extend(idx[:n_val].tolist())
        train_indices.extend(idx[n_val:].tolist())
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return np.array(train_indices, dtype=np.int64), np.array(val_indices, dtype=np.int64)


class PromptDataset(Dataset):
    def __init__(self, prompts: list[str], labels: list[str], teacher_labels: list[str | None]) -> None:
        self.prompts = prompts
        self.label_ids = [LABEL_TO_ID[label] for label in labels]
        self.direction_ids = [label_to_direction_id(label) for label in labels]
        self.teacher_ids = [LABEL_TO_ID[label] if label in LABEL_TO_ID else -100 for label in teacher_labels]

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, index: int) -> dict:
        return {
            "prompt": self.prompts[index],
            "label_id": self.label_ids[index],
            "direction_id": self.direction_ids[index],
            "teacher_id": self.teacher_ids[index],
        }


def _collate(batch: list[dict], tokenizer, max_length: int) -> dict[str, torch.Tensor]:
    prompts = [item["prompt"] for item in batch]
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    encoded["labels"] = torch.tensor([item["label_id"] for item in batch], dtype=torch.long)
    encoded["direction_labels"] = torch.tensor([item["direction_id"] for item in batch], dtype=torch.long)
    encoded["teacher_labels"] = torch.tensor([item["teacher_id"] for item in batch], dtype=torch.long)
    return encoded


def _load_official_merged_model(base_model_path: str, adapter_path: str):
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    try:
        model = model.merge_and_unload()
        print("merged LoRA into base model", flush=True)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"LoRA merge failed; compressed student must derive from merged baseline: {exc}") from exc
    model.eval()
    return model


def _build_student(args, layer_indices: list[int]) -> CompressedLlamaClassifier:
    print("loading official baseline to initialize compressed student ...", flush=True)
    official_model = _load_official_merged_model(args.base_model, args.adapter)
    print(f"creating compressed classifier from layers {layer_indices}", flush=True)
    student = create_compressed_classifier_from_causal_lm(
        official_model,
        layer_indices=layer_indices,
        max_length=args.max_length,
    )
    del official_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return student


def _make_optimizer(model: CompressedLlamaClassifier, *, lr_head: float, lr_backbone: float, weight_decay: float):
    head_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("classifier") or name.startswith("direction_classifier"):
            head_params.append(param)
        else:
            backbone_params.append(param)
    groups = []
    if head_params:
        groups.append({"params": head_params, "lr": lr_head, "weight_decay": weight_decay})
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr_backbone, "weight_decay": weight_decay})
    return torch.optim.AdamW(groups)


def _loss_fn(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], args) -> torch.Tensor:
    logits = outputs["logits"].float()
    direction_logits = outputs["direction_logits"].float()
    true_loss = F.cross_entropy(logits, batch["labels"])
    direction_loss = F.cross_entropy(direction_logits, batch["direction_labels"])
    teacher_mask = batch["teacher_labels"] >= 0
    if teacher_mask.any():
        teacher_loss = F.cross_entropy(logits[teacher_mask], batch["teacher_labels"][teacher_mask])
    else:
        teacher_loss = torch.zeros((), device=logits.device)
    return (
        args.true_loss_weight * true_loss
        + args.teacher_loss_weight * teacher_loss
        + args.direction_loss_weight * direction_loss
    )


def _evaluate(model: CompressedLlamaClassifier, loader: DataLoader, device: torch.device, name: str) -> tuple[float, float]:
    model.eval()
    y_true: list[str] = []
    y_pred: list[str] = []
    with torch.inference_mode():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pred_ids = outputs["logits"].argmax(dim=-1).cpu().tolist()
            true_ids = batch["labels"].cpu().tolist()
            y_pred.extend(ALL_LABELS[i] for i in pred_ids)
            y_true.extend(ALL_LABELS[i] for i in true_ids)
    f1 = macro_f1(y_true, y_pred)
    da = direction_acc(y_true, [label_direction(label) for label in y_pred])
    print(f"{name}: macro_f1={f1:.4f} direction_acc={da:.4f}", flush=True)
    model.train()
    return f1, da


def _checkpoint_metric(f1: float, da: float) -> tuple[int, float, float, float]:
    """Validation ordering aligned with the public score thresholds.

    First prefer models that clear both zeroing thresholds, then maximize the
    clipped W components, and finally use raw F1/DA as tie-breakers.  This keeps
    a good mid-training checkpoint from being overwritten by a worse final epoch.
    """
    clears_thresholds = int(f1 >= 0.03 and da >= 0.35)
    clipped_w = min(f1 / 0.0599, 1.0) + min(da / 0.5, 1.0)
    return clears_thresholds, clipped_w, f1, da


def _copy_state_to_cpu(model: CompressedLlamaClassifier) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def _train_once(args, prompts: list[str], labels: list[str], teacher_labels: list[str | None], layer_indices: list[int], output_dir: str | Path) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    train_idx, val_idx = _stratified_split(labels, val_ratio=args.val_ratio, seed=args.seed)
    if len(val_idx) == 0:
        train_idx = np.arange(len(prompts), dtype=np.int64)

    def subset(values, indices):
        return [values[int(i)] for i in indices]

    train_dataset = PromptDataset(subset(prompts, train_idx), subset(labels, train_idx), subset(teacher_labels, train_idx))
    val_dataset = PromptDataset(subset(prompts, val_idx), subset(labels, val_idx), subset(teacher_labels, val_idx)) if len(val_idx) else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: _collate(batch, tokenizer, args.max_length),
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda batch: _collate(batch, tokenizer, args.max_length),
        )

    student = _build_student(args, layer_indices)
    set_trainable_layers(student, unfreeze_last_layers=args.unfreeze_last_layers)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    student.to(device)
    student.train()

    optimizer = _make_optimizer(
        student,
        lr_head=args.lr_head,
        lr_backbone=args.lr_backbone,
        weight_decay=args.weight_decay,
    )

    use_amp = device.type == "cuda"
    step = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_metric: tuple[int, float, float, float] | None = None
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, args.epochs + 1):
        losses: list[float] = []
        for batch_idx, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                outputs = student(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                loss = _loss_fn(outputs, batch, args) / args.gradient_accumulation
            loss.backward()
            losses.append(float(loss.detach().cpu()) * args.gradient_accumulation)
            if batch_idx % args.gradient_accumulation == 0 or batch_idx == len(train_loader):
                torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
        print(f"epoch {epoch}/{args.epochs} loss={np.mean(losses):.4f}", flush=True)
        train_f1, train_da = _evaluate(student, train_loader, device, "train")
        if val_loader is not None:
            val_f1, val_da = _evaluate(student, val_loader, device, "val")
            metric = _checkpoint_metric(val_f1, val_da)
        else:
            metric = _checkpoint_metric(train_f1, train_da)
        if best_metric is None or metric > best_metric:
            best_metric = metric
            best_epoch = epoch
            best_state = _copy_state_to_cpu(student)
            print(
                f"new best checkpoint: epoch={best_epoch} "
                f"clears_thresholds={metric[0]} clipped_w={metric[1]:.4f} "
                f"f1={metric[2]:.4f} da={metric[3]:.4f}",
                flush=True,
            )

    if best_state is not None:
        print(f"loading best checkpoint from epoch {best_epoch} before saving", flush=True)
        student.load_state_dict(best_state)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    student.eval()
    student.to("cpu")
    student.save_classifier(
        output_dir,
        layer_indices=layer_indices,
        base_model_path=args.base_model,
        adapter_path=args.adapter,
    )
    print(f"saved compressed classifier to {output_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", nargs="+", required=True)
    parser.add_argument("--teacher-labels", default="weights/teacher_labels.jsonl")
    parser.add_argument("--output-dir", default="weights/compressed_llama_classifier")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--base-model", default="/opt/fingpt-forecaster/models/Llama-2-7b-chat-hf")
    parser.add_argument("--adapter", default="/opt/fingpt-forecaster/models/fingpt-forecaster_sz50_llama2-7B_lora")
    parser.add_argument("--kept-layers", type=int, default=8)
    parser.add_argument("--layer-indices", default="", help="Comma-separated source layer indices; overrides --kept-layers")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-backbone", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--unfreeze-last-layers", type=int, default=0)
    parser.add_argument("--true-loss-weight", type=float, default=0.7)
    parser.add_argument("--teacher-loss-weight", type=float, default=0.2)
    parser.add_argument("--direction-loss-weight", type=float, default=0.1)
    args = parser.parse_args()

    prompts, labels = _load_rows(args.parquet, args.limit)
    teacher_labels = _load_teacher_labels(args.teacher_labels, len(prompts))

    # Need source config for total layer count; load only config by peeking through AutoModel config path.
    from transformers import AutoConfig

    base_config = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True)
    layer_indices = parse_layer_indices(
        args.layer_indices,
        total_layers=int(base_config.num_hidden_layers),
        kept_layers=args.kept_layers,
    )
    print(f"loaded {len(prompts)} rows; selected layers={layer_indices}", flush=True)
    if any(label is not None for label in teacher_labels):
        print(f"teacher labels available: {sum(label is not None for label in teacher_labels)}/{len(teacher_labels)}", flush=True)
    else:
        print("teacher labels unavailable; training with gold labels and direction loss only", flush=True)

    _train_once(args, prompts, labels, teacher_labels, layer_indices, args.output_dir)


if __name__ == "__main__":
    main()
