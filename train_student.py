#!/usr/bin/env python3
"""Train the lightweight FinGPT student classifier.

Example:
  python train_student.py \
    --parquet /opt/fingpt-forecaster/datasets/fingpt-forecaster-sz50-20230201-20240101/data/train-*.parquet \
    --output weights/student_model.npz

The produced artifact is loaded by predictor.py when present.  The trainer uses
only a hashed feature extractor plus online softmax regression, so the runtime
artifact is small and does not require sklearn/joblib at evaluation time.
"""
from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path

import numpy as np

from scoring.metrics import direction_acc, macro_f1
from student_model import (
    ALL_LABELS,
    DIRECTION_LABELS,
    HashingFeatureExtractor,
    StudentTextClassifier,
    label_direction,
    save_student_artifact,
)


def _expand_paths(patterns: list[str]) -> list[str]:
    paths: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        paths.extend(matches if matches else [pattern])
    return paths


def _load_rows(patterns: list[str], limit: int) -> tuple[list[str], list[str]]:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - depends on dev environment
        raise SystemExit(
            "train_student.py needs pandas/pyarrow to read parquet files. "
            "Run it in the competition development environment, or install them "
            "outside the submitted runtime if necessary."
        ) from exc

    frames = []
    for path in _expand_paths(patterns):
        frame = pd.read_parquet(path, columns=["prompt", "label"])
        frames.append(frame)
    if not frames:
        raise SystemExit("No parquet files matched --parquet")
    df = pd.concat(frames, ignore_index=True)
    df = df[df["label"].isin(ALL_LABELS)].dropna(subset=["prompt", "label"])
    if limit > 0:
        df = df.head(limit)
    if df.empty:
        raise SystemExit("No valid rows with prompt/label were found")
    return df["prompt"].astype(str).tolist(), df["label"].astype(str).tolist()


def _stratified_split(
    labels: list[str],
    *,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
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


def _class_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    weights = np.ones(n_classes, dtype=np.float32)
    nonzero = counts > 0
    weights[nonzero] = len(y) / (n_classes * counts[nonzero])
    return np.clip(weights, 0.25, 4.0)


def _softmax(scores: np.ndarray) -> np.ndarray:
    scores = scores - np.max(scores)
    exp_scores = np.exp(scores)
    return exp_scores / np.sum(exp_scores)


def _predict_scores(
    *,
    extractor: HashingFeatureExtractor,
    weights: np.ndarray,
    bias: np.ndarray,
    prompt: str,
) -> np.ndarray:
    indices, values = extractor.vectorize(prompt)
    scores = bias.astype(np.float32, copy=True)
    if indices.size:
        scores += weights[:, indices] @ values
    return scores


def _train_softmax(
    *,
    prompts: list[str],
    y: np.ndarray,
    extractor: HashingFeatureExtractor,
    n_classes: int,
    epochs: int,
    lr: float,
    l2: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    weights = np.zeros((n_classes, extractor.dim), dtype=np.float32)
    bias = np.zeros(n_classes, dtype=np.float32)
    order = np.arange(len(prompts), dtype=np.int64)
    class_weights = _class_weights(y, n_classes)
    step = 0

    for epoch in range(1, epochs + 1):
        rng.shuffle(order)
        losses = []
        for row_index in order:
            indices, values = extractor.vectorize(prompts[int(row_index)])
            scores = bias.astype(np.float32, copy=True)
            if indices.size:
                scores += weights[:, indices] @ values
            probs = _softmax(scores)

            target = int(y[int(row_index)])
            sample_weight = float(class_weights[target])
            loss = -math.log(max(float(probs[target]), 1e-12)) * sample_weight
            losses.append(loss)

            grad = probs
            grad[target] -= 1.0
            grad *= sample_weight

            eta = lr / math.sqrt(1.0 + step / max(len(prompts), 1))
            if indices.size:
                if l2 > 0:
                    weights[:, indices] *= max(0.0, 1.0 - eta * l2)
                weights[:, indices] -= eta * grad[:, None] * values[None, :]
            bias -= eta * grad
            step += 1

        print(f"epoch {epoch}/{epochs} loss={np.mean(losses):.4f}", flush=True)
    return weights, bias


def _predict_labels(classifier: StudentTextClassifier, prompts: list[str]) -> list[str]:
    return [classifier.predict(prompt) for prompt in prompts]


def _report(name: str, y_true: list[str], y_pred: list[str]) -> None:
    pred_dirs = [label_direction(label) for label in y_pred]
    print(
        f"{name}: macro_f1={macro_f1(y_true, y_pred):.4f} "
        f"direction_acc={direction_acc(y_true, pred_dirs):.4f}",
        flush=True,
    )


def _select_direction_weight(
    *,
    extractor: HashingFeatureExtractor,
    label_weights: np.ndarray,
    label_bias: np.ndarray,
    direction_weights: np.ndarray,
    direction_bias: np.ndarray,
    labels: tuple[str, ...],
    prompts: list[str],
    y_true: list[str],
) -> float:
    if not prompts:
        return 0.35
    best_weight = 0.0
    best_score = -1.0
    for candidate in (0.0, 0.15, 0.3, 0.5, 0.8, 1.2):
        classifier = StudentTextClassifier(
            extractor=extractor,
            weights=label_weights,
            bias=label_bias,
            labels=labels,
            direction_weights=direction_weights,
            direction_bias=direction_bias,
            direction_weight=candidate,
        )
        pred = _predict_labels(classifier, prompts)
        f1 = macro_f1(y_true, pred)
        da = direction_acc(y_true, [label_direction(label) for label in pred])
        score = min(f1 / 0.0599, 1.0) + min(da / 0.5, 1.0)
        if score > best_score:
            best_score = score
            best_weight = candidate
    return best_weight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", nargs="+", required=True, help="Training parquet path/glob")
    parser.add_argument("--output", default="weights/student_model.npz")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dim", type=int, default=1 << 18)
    parser.add_argument("--ngram-min", type=int, default=2)
    parser.add_argument("--ngram-max", type=int, default=4)
    parser.add_argument("--max-chars", type=int, default=12000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.35)
    parser.add_argument("--l2", type=float, default=1e-6)
    args = parser.parse_args()

    prompts, labels = _load_rows(args.parquet, args.limit)
    label_to_id = {label: i for i, label in enumerate(ALL_LABELS)}
    direction_to_id = {direction: i for i, direction in enumerate(DIRECTION_LABELS)}
    y_all = np.array([label_to_id[label] for label in labels], dtype=np.int64)
    y_dir_all = np.array(
        [direction_to_id[label_direction(label)] for label in labels],
        dtype=np.int64,
    )

    train_idx, val_idx = _stratified_split(labels, val_ratio=args.val_ratio, seed=args.seed)
    train_prompts = [prompts[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    train_y = y_all[train_idx]
    train_y_dir = y_dir_all[train_idx]
    val_prompts = [prompts[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    print(
        f"loaded {len(prompts)} rows; train={len(train_prompts)} val={len(val_prompts)}",
        flush=True,
    )

    extractor = HashingFeatureExtractor(
        dim=args.dim,
        ngram_min=args.ngram_min,
        ngram_max=args.ngram_max,
        max_chars=args.max_chars,
    )

    print("training 13-class classifier ...", flush=True)
    weights, bias = _train_softmax(
        prompts=train_prompts,
        y=train_y,
        extractor=extractor,
        n_classes=len(ALL_LABELS),
        epochs=args.epochs,
        lr=args.lr,
        l2=args.l2,
        seed=args.seed,
    )

    print("training direction auxiliary classifier ...", flush=True)
    direction_weights, direction_bias = _train_softmax(
        prompts=train_prompts,
        y=train_y_dir,
        extractor=extractor,
        n_classes=len(DIRECTION_LABELS),
        epochs=max(3, args.epochs // 2),
        lr=args.lr,
        l2=args.l2,
        seed=args.seed + 1,
    )

    direction_weight = _select_direction_weight(
        extractor=extractor,
        label_weights=weights,
        label_bias=bias,
        direction_weights=direction_weights,
        direction_bias=direction_bias,
        labels=ALL_LABELS,
        prompts=val_prompts or train_prompts,
        y_true=val_labels or train_labels,
    )
    print(f"selected direction_weight={direction_weight}", flush=True)

    classifier = StudentTextClassifier(
        extractor=extractor,
        weights=weights,
        bias=bias,
        labels=ALL_LABELS,
        direction_weights=direction_weights,
        direction_bias=direction_bias,
        direction_weight=direction_weight,
    )

    _report("train", train_labels, _predict_labels(classifier, train_prompts))
    if val_prompts:
        _report("val", val_labels, _predict_labels(classifier, val_prompts))

    save_student_artifact(Path(args.output), classifier=classifier)
    print(f"saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
