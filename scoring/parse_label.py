"""从模型生成文本中解析预测标签。

解析规则（公开给参赛队，保证判分口径无争议）：

1. 在文本中查找所有"预测涨跌幅：xxx"行（冒号可全角/半角/省略），
   对每行内容依次尝试：
   a) 13 类区间标签模式（如"上涨1-2%"），取位置最靠前者；
   b) 数值分箱：如"下跌0.5%"按区间归类（0<x≤1→0-1%，1<x≤2→1-2%，…，
      x>5→超过5%，x=0→股价持平）。模型输出自由数值是基线模型的真实行为，
      赛题文档"将模型输出映射到12个细粒度类别"即指此映射。
   取第一行能解析出合法标签的结果。
2. 若没有任何"预测涨跌幅"行能解析出标签，则在全文中扫描 13 类标签模式
   （仅区间标签，不做数值分箱——分析正文里常出现历史涨跌数值）：
   - 全文只出现一种标签 → 取该标签（兼容 predict 直接返回裸标签的提交）；
   - 出现多种不同标签或一种都没有 → 解析失败（该样本 0 分）。
3. 匹配前做归一化：全角％/：转半角，~ ～ – — − 等连接符统一为 -，
   去除空白与 markdown 加粗符号。

同一行出现多个标签时取位置最靠前者。
"""
from __future__ import annotations

import math
import re

from .labels import FLAT_LABEL, label_direction

# 归一化映射
_NORMALIZE_TABLE = str.maketrans({
    "％": "%",
    "：": ":",
    "～": "-",
    "~": "-",
    "–": "-",
    "—": "-",
    "−": "-",
    "－": "-",
    "，": ",",
})

# (canonical_label, 编译后的模式)。模式在归一化后的文本上匹配。
_LABEL_PATTERNS: list[tuple[str, re.Pattern]] = []


def _build_patterns() -> None:
    for direction in ("上涨", "下跌"):
        for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5)):
            canonical = f"{direction}{lo}-{hi}%"
            _LABEL_PATTERNS.append(
                (canonical, re.compile(rf"{direction}\s*{lo}\s*-\s*{hi}\s*%"))
            )
        canonical = f"{direction}超过5%"
        _LABEL_PATTERNS.append(
            (canonical, re.compile(rf"{direction}\s*(?:超过\s*5\s*%|5\s*%\s*以上)"))
        )
    _LABEL_PATTERNS.append((FLAT_LABEL, re.compile(r"(?:股价)?持平")))


_build_patterns()

_PREDICTION_LINE = re.compile(r"预测涨跌幅\s*:?\s*([^\n]*)")

# 自由数值区间形式（13 类外），如"上涨5-6%"、"下跌5~7%"（归一化后连接符已统一为 -）
_NUMERIC_RANGE = re.compile(
    r"(上涨|下跌)(?:约|大约|接近|近)?(\d+(?:\.\d+)?)\s*[-到至]\s*(\d+(?:\.\d+)?)\s*%"
)
# 自由数值形式，如"下跌0.5%"、"上涨约2.3%"
_NUMERIC = re.compile(r"(上涨|下跌)(?:约|大约|接近|近)?(\d+(?:\.\d+)?)\s*%")


def _normalize(text: str) -> str:
    text = text.translate(_NORMALIZE_TABLE)
    return text.replace("**", "").replace(" ", "")


def _first_label(text: str) -> str | None:
    """返回文本中位置最靠前的标签匹配，无匹配返回 None。"""
    best: tuple[int, str] | None = None
    for label, pattern in _LABEL_PATTERNS:
        m = pattern.search(text)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), label)
    return best[1] if best else None


def _all_labels(text: str) -> set[str]:
    return {label for label, pattern in _LABEL_PATTERNS if pattern.search(text)}


def _numeric_label(text: str) -> str | None:
    """数值分箱：0<x≤1→0-1%，…，4<x≤5→4-5%，x>5→超过5%，x=0→持平。

    自由数值区间（13 类外，如"5-6%"）按区间中点归箱：5-6%→中点5.5>5→超过5%；
    "2-4%"→中点3→2-3%。区间优先于单值匹配（避免只取到区间左端点）。
    """
    m = _NUMERIC_RANGE.search(text)
    if m:
        direction = m.group(1)
        value = (float(m.group(2)) + float(m.group(3))) / 2
    else:
        m = _NUMERIC.search(text)
        if not m:
            return None
        direction, value = m.group(1), float(m.group(2))
    if value == 0:
        return FLAT_LABEL
    if value > 5:
        return f"{direction}超过5%"
    lo = math.ceil(value) - 1
    return f"{direction}{lo}-{lo + 1}%"


def parse_prediction(text: str) -> str | None:
    """从生成文本解析预测标签，失败返回 None。"""
    if not text:
        return None
    normalized = _normalize(text)

    # 规则 1：优先解析"预测涨跌幅"行（区间标签 → 数值分箱）
    for m in _PREDICTION_LINE.finditer(normalized):
        label = _first_label(m.group(1)) or _numeric_label(m.group(1))
        if label:
            return label

    # 规则 2：全文唯一标签兜底（裸标签提交）
    found = _all_labels(normalized)
    if len(found) == 1:
        return found.pop()
    return None


# 方向词（仅用于在"预测涨跌幅"行里抠方向；模型只给方向、未给幅度时仍可评方向分）
_DIR_UP = re.compile(r"上涨|上升|看涨|走高|回升|攀升|增长|看多")
_DIR_DOWN = re.compile(r"下跌|下降|看跌|走低|回落|下挫|跌落|看空")
_DIR_FLAT = re.compile(r"持平|稳定|维持|不变|横盘|波动不大")


def _line_direction(content: str) -> str | None:
    """从一行内容里取位置最靠前的方向词，映射为 up/down/flat。"""
    best: tuple[int, str] | None = None
    for d, pat in (("up", _DIR_UP), ("down", _DIR_DOWN), ("flat", _DIR_FLAT)):
        m = pat.search(content)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), d)
    return best[1] if best else None


def parse_direction(text: str) -> str | None:
    """方向解析（涨/跌/平），失败返回 None。

    口径：先按 parse_prediction 抠出 13 类标签并取其方向（与 F1 完全一致）；
    若标签解析失败（模型只写了"预测涨跌幅：上涨"没给幅度、或给了 13 类外的
    幅度如"5-6%"），再退一步只从"预测涨跌幅"行里抠方向词。仅扫预测行、不扫
    分析正文（正文里涨跌字眼是历史数据，会误判）。
    """
    label = parse_prediction(text)
    if label:
        return label_direction(label)
    if not text:
        return None
    normalized = _normalize(text)
    for m in _PREDICTION_LINE.finditer(normalized):
        d = _line_direction(m.group(1))
        if d:
            return d
    return None
