"""参赛提交入口：Predictor 类。

判题系统按如下契约调用，请勿改变类名与方法签名：

    p = Predictor()
    p.load()
    for prompt in 评测集:
        text = p.predict(prompt)

本实现优先使用基于官方 FinGPT 基线大模型结构压缩得到的 Llama 分类器；若压缩
模型目录不存在或加载失败，则回退到官方 FinGPT 7B + LoRA 保底模型。
"""
from __future__ import annotations

import os
import re
from pathlib import Path


class Predictor:
    def __init__(self):
        root = Path(__file__).resolve().parent
        compressed_dir = os.environ.get("COMPRESSED_LLAMA_DIR")
        if compressed_dir:
            compressed_path = Path(compressed_dir)
            self.compressed_dir = compressed_path if compressed_path.is_absolute() else root / compressed_path
        else:
            self.compressed_dir = root / "weights" / "compressed_llama_classifier"
        self.model = None
        self.tokenizer = None
        self.mode = "unloaded"
        self.base_model_path = "/opt/fingpt-forecaster/models/Llama-2-7b-chat-hf"
        self.adapter_path = "/opt/fingpt-forecaster/models/fingpt-forecaster_sz50_llama2-7B_lora"

    def load(self):
        """Load compressed baseline-derived classifier when available, otherwise fallback."""
        if (self.compressed_dir / "classifier_config.json").exists():
            try:
                self._load_compressed_classifier()
                return
            except Exception as exc:  # noqa: BLE001 - fallback keeps the submission runnable.
                print(f"Compressed classifier load failed ({type(exc).__name__}: {exc}); falling back.", flush=True)
                self.model = None
                self.tokenizer = None

        self._load_teacher_model()

    def _load_compressed_classifier(self) -> None:
        import torch
        from transformers import AutoTokenizer

        from compressed_llama_classifier import CompressedLlamaClassifier

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path, trust_remote_code=True)
        self.tokenizer.truncation_side = "left"
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = CompressedLlamaClassifier.load_classifier(self.compressed_dir, dtype=dtype)
        self.model.to(device)
        self.model.eval()
        self.mode = "compressed"
        print(f"Loaded compressed Llama classifier: {self.compressed_dir}", flush=True)

    def _load_teacher_model(self) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"正在加载官方保底模型...\n基础模型: {self.base_model_path}", flush=True)
        print(f"LoRA适配器: {self.adapter_path}", flush=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",
        )
        base_model.eval()
        self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
        self.model.eval()
        self.mode = "teacher"
        print("【成功】官方 FinGPT 保底模型加载完成。", flush=True)

    def _format_prediction(self, raw_output: str) -> str:
        raw_text = raw_output.strip()
        direction = "股价持平"

        if "上涨" in raw_text or "涨" in raw_text:
            match = re.search(r"(\d+(?:\.\d+)?)%", raw_text)
            if match:
                num = float(match.group(1))
                if num > 5:
                    direction = "上涨超过5%"
                elif num > 4:
                    direction = "上涨4-5%"
                elif num > 3:
                    direction = "上涨3-4%"
                elif num > 2:
                    direction = "上涨2-3%"
                elif num > 1:
                    direction = "上涨1-2%"
                else:
                    direction = "上涨0-1%"
            else:
                direction = "上涨0-1%"
        elif "下跌" in raw_text or "跌" in raw_text:
            match = re.search(r"(\d+(?:\.\d+)?)%", raw_text)
            if match:
                num = float(match.group(1))
                if num > 5:
                    direction = "下跌超过5%"
                elif num > 4:
                    direction = "下跌4-5%"
                elif num > 3:
                    direction = "下跌3-4%"
                elif num > 2:
                    direction = "下跌2-3%"
                elif num > 1:
                    direction = "下跌1-2%"
                else:
                    direction = "下跌0-1%"
            else:
                direction = "下跌0-1%"

        return f"预测涨跌幅：{direction}"

    def _predict_compressed(self, prompt: str) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("压缩模型未加载，请先调用 load() 方法")

        import torch

        device = next(self.model.parameters()).device
        model_max_length = int(getattr(self.model, "max_length", 4096))
        env_max_length = os.environ.get("COMPRESSED_LLAMA_MAX_LENGTH")
        if env_max_length:
            max_length = min(model_max_length, int(env_max_length))
        else:
            max_length = min(model_max_length, 2048)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(device)
        use_amp = device.type == "cuda"
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            label = self.model.predict_label(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
            )
        return f"预测涨跌幅：{label}"

    def _predict_teacher(self, prompt: str) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("模型未加载，请先调用 load() 方法")

        import torch

        device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        input_len = int(inputs["input_ids"].shape[1])

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=30,
                do_sample=False,
                top_p=None,
                temperature=None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = generated_ids[0, input_len:]
        output_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return self._format_prediction(output_text)

    def predict(self, prompt: str) -> str:
        if self.mode == "compressed":
            return self._predict_compressed(prompt)
        if self.mode == "teacher":
            return self._predict_teacher(prompt)
        raise RuntimeError("模型未加载，请先调用 load() 方法")
