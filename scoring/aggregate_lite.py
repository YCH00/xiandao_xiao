"""学生本地自测用的轻量评分入口（local_eval.py 调用）。

与平台评分共用 parse_label / metrics，仅省去 marker 交叉校验等判题机制。
该文件随 scoring/ 一起拷贝进学生模板（scripts/sync_template_scoring.sh）。
"""
from __future__ import annotations

from .metrics import compute_scores, direction_acc, macro_f1
from .parse_label import parse_direction, parse_prediction


def summarize(
    records: list[dict],
    *,
    vram_mb: float,
    baseline: dict,
    f1_zero_threshold: float,
    da_zero_threshold: float,
) -> dict:
    """records: [{"text", "status", "latency_s", "label"}, ...]"""
    y_true = [r["label"] for r in records]
    y_pred = [
        parse_prediction(r["text"]) if r["status"] == "ok" else None for r in records
    ]
    y_pred_dir = [
        parse_direction(r["text"]) if r["status"] == "ok" else None for r in records
    ]
    latencies = [r["latency_s"] for r in records]
    f1 = macro_f1(y_true, y_pred)
    da = direction_acc(y_true, y_pred_dir)
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    scores = compute_scores(
        vram_mb=vram_mb,
        avg_latency_s=avg_latency,
        f1=f1,
        da=da,
        baseline_vram_mb=baseline["vram_mb"],
        baseline_latency_s=baseline["avg_latency_s"],
        baseline_f1=baseline["macro_f1"],
        baseline_da=baseline["direction_acc"],
        f1_zero_threshold=f1_zero_threshold,
        da_zero_threshold=da_zero_threshold,
    )
    return {
        "n": len(records),
        "macro_f1": round(f1, 4),
        "direction_acc": round(da, 4),
        "avg_latency_s": round(avg_latency, 2),
        "vram_mb": round(vram_mb, 1),
        "n_parsed": sum(1 for p in y_pred if p is not None),
        "n_direction_scored": sum(1 for d in y_pred_dir if d is not None),
        "score_u": scores.score_u,
        "score_v": scores.score_v,
        "score_w": scores.score_w,
        "score_total": scores.score_total,
        "zeroed_by_f1": scores.zeroed_by_f1,
    }
