#!/usr/bin/env python3
"""本地自测脚本：用公开测试集评估你的 Predictor，估算 U/V/W 分数。

与判题系统使用完全相同的解析与评分代码（scoring/ 目录），提交前先在
自己服务器上跑通本脚本，可避免浪费每日提交次数。

用法（在 submission 目录内）：
  uv run python local_eval.py --parquet <公开测试集test.parquet> --limit 20 [--gpu 0]

注意：正式判题用的是隐藏测试集的 100 条分层子集，本地分数仅供参考。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scoring.aggregate_lite import summarize  # noqa: E402

# 与 infra/config.json 公布值保持一致（基线常量正式值以平台公告为准）
BASELINE = {"vram_mb": 20165, "avg_latency_s": 105.65, "macro_f1": 0.0599,
            "direction_acc": 0.5}
F1_ZERO_THRESHOLD = 0.03
DA_ZERO_THRESHOLD = 0.35


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, help="公开测试集 test parquet")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--debug-errors", action="store_true", help="print the first prediction traceback")
    args = ap.parse_args()

    os.environ.setdefault("HIP_VISIBLE_DEVICES", str(args.gpu))
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))

    import pandas as pd  # noqa: PLC0415

    df = pd.read_parquet(args.parquet)
    if args.limit > 0:
        df = df.head(args.limit)

    from predictor import Predictor  # noqa: PLC0415

    p = Predictor()
    print("loading model ...", flush=True)
    p.load()

    import torch  # noqa: PLC0415

    torch.cuda.reset_peak_memory_stats()
    records = []
    print(f"starting evaluation on {len(df)} rows ...", flush=True)
    for i, row in enumerate(df.itertuples(index=False)):
        t0 = time.perf_counter()
        try:
            if args.debug_errors:
                print(f"predicting row {i + 1}/{len(df)} ...", flush=True)
            text = p.predict(row.prompt)
            status = "ok"
        except Exception as e:  # noqa: BLE001
            text, status = "", f"predict_error:{type(e).__name__}"
            if args.debug_errors:
                print(f"prediction failed on row {i + 1}: {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
                args.debug_errors = False
        latency = time.perf_counter() - t0
        records.append({"text": text, "status": status, "latency_s": latency, "label": row.label})
        print(f"[{i + 1}/{len(df)}] {latency:.1f}s {status}", flush=True)

    # 本地显存参考值用 torch 统计（正式判题为容器外 rocm-smi 进程级峰值，略有差异）
    vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    report = summarize(records, vram_mb=vram_mb, baseline=BASELINE,
                       f1_zero_threshold=F1_ZERO_THRESHOLD,
                       da_zero_threshold=DA_ZERO_THRESHOLD)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
