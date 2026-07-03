"""Lightweight text classifier used by the submission predictor.

The runtime path intentionally depends only on the Python standard library and
numpy.  Training code lives in train_student.py; this module is shared by the
trainer and Predictor so the feature extraction used offline is byte-for-byte
the same as the one used during evaluation.
"""
from __future__ import annotations

import math
import re
import zlib
from pathlib import Path

import numpy as np


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

_NORMALIZE_TABLE = str.maketrans(
    {
        "％": "%",
        "：": ":",
        "～": "-",
        "~": "-",
        "–": "-",
        "—": "-",
        "−": "-",
        "－": "-",
        "，": ",",
    }
)

_PERCENT_RE = re.compile(r"([+-]?\d+(?:\.\d+)?)\s*%")
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9])([+-]?\d+(?:\.\d+)?)(?![A-Za-z0-9])")

_KEYWORDS = (
    "上涨",
    "上升",
    "走高",
    "看涨",
    "增长",
    "涨停",
    "反弹",
    "下跌",
    "下降",
    "走低",
    "看跌",
    "回落",
    "跌停",
    "持平",
    "震荡",
    "横盘",
    "波动",
    "成交量",
    "换手率",
    "净利润",
    "营收",
    "公告",
    "研报",
    "资金",
    "主力",
    "北向",
)


def label_direction(label: str) -> str:
    if label.startswith("上涨"):
        return "up"
    if label.startswith("下跌"):
        return "down"
    return "flat"


def _bucket_number(value: float) -> str:
    abs_value = abs(value)
    if abs_value == 0:
        bucket = "0"
    elif abs_value <= 1:
        bucket = "0-1"
    elif abs_value <= 2:
        bucket = "1-2"
    elif abs_value <= 3:
        bucket = "2-3"
    elif abs_value <= 4:
        bucket = "3-4"
    elif abs_value <= 5:
        bucket = "4-5"
    elif abs_value <= 10:
        bucket = "5-10"
    else:
        bucket = "10+"
    if value > 0:
        sign = "pos"
    elif value < 0:
        sign = "neg"
    else:
        sign = "zero"
    return f"{sign}:{bucket}"


class HashingFeatureExtractor:
    """Stateless hashed character n-gram extractor.

    Character n-grams are robust for the mixed Chinese/English/numeric prompts
    used by FinGPT.  A small set of numeric buckets and finance keywords is
    added to make the linear classifier less dependent on exact phrasing.
    """

    def __init__(
        self,
        *,
        dim: int = 1 << 18,
        ngram_min: int = 2,
        ngram_max: int = 4,
        max_chars: int = 12000,
    ) -> None:
        if dim <= 0 or dim & (dim - 1) != 0:
            raise ValueError("dim must be a positive power of two")
        if ngram_min <= 0 or ngram_max < ngram_min:
            raise ValueError("invalid n-gram range")
        self.dim = int(dim)
        self.ngram_min = int(ngram_min)
        self.ngram_max = int(ngram_max)
        self.max_chars = int(max_chars)

    def _normalize(self, text: str) -> str:
        text = text.translate(_NORMALIZE_TABLE).lower()
        text = re.sub(r"\s+", "", text)
        if len(text) <= self.max_chars:
            return text
        head_len = self.max_chars // 3
        tail_len = self.max_chars - head_len
        return text[:head_len] + text[-tail_len:]

    def _hash(self, token: str) -> tuple[int, float]:
        h = zlib.crc32(token.encode("utf-8")) & 0xFFFFFFFF
        index = h & (self.dim - 1)
        sign = 1.0 if (h & 0x80000000) == 0 else -1.0
        return index, sign

    def _iter_tokens(self, text: str) -> tuple[str, ...]:
        normalized = self._normalize(text)
        tokens: list[str] = []

        for n in range(self.ngram_min, self.ngram_max + 1):
            if len(normalized) < n:
                continue
            prefix = f"c{n}:"
            tokens.extend(prefix + normalized[i : i + n] for i in range(len(normalized) - n + 1))

        for keyword in _KEYWORDS:
            count = normalized.count(keyword)
            if count:
                tokens.append(f"kw:{keyword}")
                tokens.append(f"kwc:{keyword}:{min(count, 5)}")

        for match in _PERCENT_RE.finditer(normalized):
            value = float(match.group(1))
            tokens.append(f"pct:{_bucket_number(value)}")

        # Keep only coarse numeric information so exact dates/prices do not
        # dominate the representation.
        for match in _NUMBER_RE.finditer(normalized):
            value = float(match.group(1))
            if abs(value) > 100000:
                continue
            tokens.append(f"num:{_bucket_number(value)}")

        return tuple(tokens)

    def vectorize(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        values_by_index: dict[int, float] = {}
        for token in self._iter_tokens(text):
            index, sign = self._hash(token)
            values_by_index[index] = values_by_index.get(index, 0.0) + sign

        if not values_by_index:
            return (
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.float32),
            )

        indices = np.fromiter(values_by_index.keys(), dtype=np.int64)
        values = np.fromiter(values_by_index.values(), dtype=np.float32)
        norm = math.sqrt(float(np.dot(values, values)))
        if norm > 0:
            values /= norm
        return indices, values


class StudentTextClassifier:
    def __init__(
        self,
        *,
        extractor: HashingFeatureExtractor,
        weights: np.ndarray,
        bias: np.ndarray,
        labels: tuple[str, ...] = ALL_LABELS,
        direction_weights: np.ndarray | None = None,
        direction_bias: np.ndarray | None = None,
        direction_weight: float = 0.0,
    ) -> None:
        self.extractor = extractor
        self.weights = np.asarray(weights, dtype=np.float32)
        self.bias = np.asarray(bias, dtype=np.float32)
        self.labels = tuple(labels)
        self.direction_weights = (
            None if direction_weights is None else np.asarray(direction_weights, dtype=np.float32)
        )
        self.direction_bias = (
            None if direction_bias is None else np.asarray(direction_bias, dtype=np.float32)
        )
        self.direction_weight = float(direction_weight)
        self._label_direction_indices = np.array(
            [DIRECTION_LABELS.index(label_direction(label)) for label in self.labels],
            dtype=np.int64,
        )

        if self.weights.shape != (len(self.labels), self.extractor.dim):
            raise ValueError(
                f"weights shape {self.weights.shape} does not match "
                f"({len(self.labels)}, {self.extractor.dim})"
            )
        if self.bias.shape != (len(self.labels),):
            raise ValueError(f"bias shape {self.bias.shape} does not match ({len(self.labels)},)")
        if (self.direction_weights is None) != (self.direction_bias is None):
            raise ValueError("direction_weights and direction_bias must be provided together")
        if self.direction_weights is not None:
            if self.direction_weights.shape != (len(DIRECTION_LABELS), self.extractor.dim):
                raise ValueError("invalid direction_weights shape")
            if self.direction_bias.shape != (len(DIRECTION_LABELS),):
                raise ValueError("invalid direction_bias shape")

    @classmethod
    def load(cls, path: str | Path) -> "StudentTextClassifier":
        data = np.load(Path(path), allow_pickle=False)
        dim = int(data["dim"][0])
        extractor = HashingFeatureExtractor(
            dim=dim,
            ngram_min=int(data["ngram_min"][0]),
            ngram_max=int(data["ngram_max"][0]),
            max_chars=int(data["max_chars"][0]),
        )
        direction_weights = data["direction_weights"] if "direction_weights" in data else None
        direction_bias = data["direction_bias"] if "direction_bias" in data else None
        direction_weight = float(data["direction_weight"][0]) if "direction_weight" in data else 0.0
        labels = tuple(str(x) for x in data["labels"].tolist())
        return cls(
            extractor=extractor,
            weights=data["weights"],
            bias=data["bias"],
            labels=labels,
            direction_weights=direction_weights,
            direction_bias=direction_bias,
            direction_weight=direction_weight,
        )

    def scores(self, prompt: str) -> np.ndarray:
        indices, values = self.extractor.vectorize(prompt)
        scores = self.bias.astype(np.float32, copy=True)
        if indices.size:
            scores += self.weights[:, indices] @ values
        if (
            self.direction_weight
            and self.direction_weights is not None
            and self.direction_bias is not None
        ):
            direction_scores = self.direction_bias.astype(np.float32, copy=True)
            if indices.size:
                direction_scores += self.direction_weights[:, indices] @ values
            scores += self.direction_weight * direction_scores[self._label_direction_indices]
        return scores

    def predict(self, prompt: str) -> str:
        scores = self.scores(prompt)
        return self.labels[int(np.argmax(scores))]


def save_student_artifact(
    path: str | Path,
    *,
    classifier: StudentTextClassifier,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "weights": classifier.weights.astype(np.float32),
        "bias": classifier.bias.astype(np.float32),
        "labels": np.array(classifier.labels),
        "dim": np.array([classifier.extractor.dim], dtype=np.int64),
        "ngram_min": np.array([classifier.extractor.ngram_min], dtype=np.int64),
        "ngram_max": np.array([classifier.extractor.ngram_max], dtype=np.int64),
        "max_chars": np.array([classifier.extractor.max_chars], dtype=np.int64),
        "direction_weight": np.array([classifier.direction_weight], dtype=np.float32),
    }
    if classifier.direction_weights is not None and classifier.direction_bias is not None:
        arrays["direction_weights"] = classifier.direction_weights.astype(np.float32)
        arrays["direction_bias"] = classifier.direction_bias.astype(np.float32)
    np.savez_compressed(path, **arrays)
