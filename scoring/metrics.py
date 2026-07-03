"""宏 F1、方向准确率与 U/V/W 总分计算。

口径（公开给参赛队）：
- 宏 F1：按评测集中真实出现的标签类别（评测子集分层抽样保证 13 类齐全）
  逐类计算 F1 后取算术平均；解析失败/异常样本视为预测错误（无预测类别）。
- DirectionAcc：方向准确率。13 类映射到 涨/跌/平 三个方向，
  预测方向与真实方向一致的样本占比；解析失败/异常样本算错。
- U = max(1 - 学生显存 / 基线显存, 0) × 40
- V = max(1 - 学生平均样本时长 / 基线平均样本时长, 0) × 20
- W = [min(学生F1 / 基线F1, 1) + min(学生DirectionAcc / 基线DirectionAcc, 1)] × 20
- 若学生宏 F1 低于 f1_zero_threshold 或 DirectionAcc 低于 da_zero_threshold，
  总分记 0（拦截"假模型/乱输出"刷 U/V 分；阈值设在随机水平之下，
  不影响任何真实推理的提交）。
"""
from __future__ import annotations

from dataclasses import dataclass

from .labels import label_direction


def macro_f1(y_true: list[str], y_pred: list[str | None]) -> float:
    """13 类宏 F1。y_pred 中 None 表示该样本无有效预测。"""
    if not y_true:
        return 0.0
    classes = sorted(set(y_true))
    f1_sum = 0.0
    for c in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        denom = 2 * tp + fp + fn
        f1_sum += (2 * tp / denom) if denom else 0.0
    return f1_sum / len(classes)


def direction_acc(y_true: list[str], y_pred_dir: list[str | None]) -> float:
    """方向准确率（涨/跌/平）。

    y_pred_dir 为各样本的预测方向（parse_direction 的结果：up/down/flat 或 None）。
    None 表示连方向都判不出（算错）。注意第二个参数是"方向"不是"标签"——
    这样模型只给方向、未给幅度时仍能在方向准确率上得分（F1 仍需完整标签）。
    """
    if not y_true:
        return 0.0
    hit = sum(
        1 for t, pd in zip(y_true, y_pred_dir)
        if pd is not None and label_direction(t) == pd
    )
    return hit / len(y_true)


@dataclass
class ScoreResult:
    score_u: float
    score_v: float
    score_w: float
    score_total: float
    zeroed_by_f1: bool


def compute_scores(
    *,
    vram_mb: float,
    avg_latency_s: float,
    f1: float,
    da: float,
    baseline_vram_mb: float,
    baseline_latency_s: float,
    baseline_f1: float,
    baseline_da: float,
    f1_zero_threshold: float,
    da_zero_threshold: float,
) -> ScoreResult:
    u = max(1 - vram_mb / baseline_vram_mb, 0.0) * 40 if baseline_vram_mb > 0 else 0.0
    v = max(1 - avg_latency_s / baseline_latency_s, 0.0) * 20 if baseline_latency_s > 0 else 0.0
    w_f1 = min(f1 / baseline_f1, 1.0) if baseline_f1 > 0 else 0.0
    w_da = min(da / baseline_da, 1.0) if baseline_da > 0 else 0.0
    w = (w_f1 + w_da) * 20

    zeroed = f1 < f1_zero_threshold or da < da_zero_threshold
    total = 0.0 if zeroed else u + v + w
    return ScoreResult(
        score_u=round(u, 2),
        score_v=round(v, 2),
        score_w=round(w, 2),
        score_total=round(total, 2),
        zeroed_by_f1=zeroed,
    )
