#!/usr/bin/env python3
"""Generate teacher labels from the official FinGPT baseline model.

The output JSONL is used by train_student.py for hard-label distillation.  The
teacher is the required baseline route: Llama2-7B + FinGPT LoRA.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from scoring.parse_label import parse_prediction


def _expand_paths(patterns: list[str]) -> list[str]:
    paths: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        paths.extend(matches)
    if not paths:
        raise SystemExit(f"No parquet files matched: {patterns}")
    return list(dict.fromkeys(paths))


def _load_rows(patterns: list[str], limit: int):
    import pandas as pd

    frames = [pd.read_parquet(path, columns=["prompt", "label"]) for path in _expand_paths(patterns)]
    df = pd.concat(frames, ignore_index=True).dropna(subset=["prompt", "label"])
    if limit > 0:
        df = df.head(limit)
    return df["prompt"].astype(str).tolist(), df["label"].astype(str).tolist()


def _format_prediction(raw_output: str) -> str:
    text = raw_output.strip()
    # Keep the teacher output parseable even when it emits free-form analysis.
    label = parse_prediction(text)
    if label is not None:
        return f"预测涨跌幅：{label}"
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", nargs="+", required=True)
    parser.add_argument("--output", default="weights/teacher_labels.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--base-model", default="/opt/fingpt-forecaster/models/Llama-2-7b-chat-hf")
    parser.add_argument("--adapter", default="/opt/fingpt-forecaster/models/fingpt-forecaster_sz50_llama2-7B_lora")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--debug-first", type=int, default=3, help="Print raw teacher generations for the first N rows")
    args = parser.parse_args()

    prompts, labels = _load_rows(args.parquet, args.limit)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(base_model, args.adapter)
    model.eval()
    device = next(model.parameters()).device

    parsed = 0
    source_counts: dict[str, int] = {}
    with output_path.open("w", encoding="utf-8") as f:
        for i, (prompt, gold_label) in enumerate(zip(prompts, labels)):
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_len = int(inputs["input_ids"].shape[1])
            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    top_p=None,
                    temperature=None,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            new_tokens = generated_ids[0, input_len:]
            raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            full_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            teacher_label = parse_prediction(_format_prediction(raw_text))
            parse_source = "new_tokens"
            if teacher_label is None:
                teacher_label = parse_prediction(full_text)
                parse_source = "full_text"
            if teacher_label is None:
                # The official generator occasionally emits no parseable label under
                # short deterministic decoding.  Keep training useful by falling
                # back to the public gold label while recording that fallback.
                teacher_label = gold_label
                parse_source = "gold_fallback"
            parsed += int(teacher_label is not None)
            source_counts[parse_source] = source_counts.get(parse_source, 0) + 1
            if i < args.debug_first:
                print("--- teacher raw output ---", flush=True)
                print(raw_text[:1000], flush=True)
                print(f"parse_source={parse_source}", flush=True)
            record = {
                "index": i,
                "prompt": prompt,
                "label": gold_label,
                "teacher_text": raw_text,
                "teacher_label": teacher_label,
                "teacher_parse_source": parse_source,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[{i + 1}/{len(prompts)}] teacher_label={teacher_label}", flush=True)

    print(f"saved {output_path}; parsed={parsed}/{len(prompts)}; sources={source_counts}", flush=True)


if __name__ == "__main__":
    main()
