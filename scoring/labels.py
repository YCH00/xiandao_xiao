"""13 类标签体系定义。

12 档涨跌区间（赛题文档的 U1-U5+/D1-D5+）+ 数据集中实际存在的"股价持平"，
与 FinGPT/fingpt-forecaster-sz50 数据集的 label 字段取值完全一致。
"""

UP_LABELS = [
    "上涨0-1%",
    "上涨1-2%",
    "上涨2-3%",
    "上涨3-4%",
    "上涨4-5%",
    "上涨超过5%",
]

DOWN_LABELS = [
    "下跌0-1%",
    "下跌1-2%",
    "下跌2-3%",
    "下跌3-4%",
    "下跌4-5%",
    "下跌超过5%",
]

FLAT_LABEL = "股价持平"

ALL_LABELS = UP_LABELS + DOWN_LABELS + [FLAT_LABEL]

assert len(ALL_LABELS) == 13


def label_direction(label: str | None) -> str | None:
    """13 类标签 → 方向（涨/跌/平）。None → None。"""
    if label is None:
        return None
    if label in UP_LABELS:
        return "up"
    if label in DOWN_LABELS:
        return "down"
    return "flat"
