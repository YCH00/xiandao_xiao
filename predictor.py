"""参赛提交入口：Predictor 类。

判题系统按如下契约调用，请勿改变类名与方法签名：

    p = Predictor()
    p.load()
    for prompt in 评测集:
        text = p.predict(prompt)

本实现优先使用 weights/student_model.npz 中的轻量 13 类分类器；若该文件不存在
或加载失败，则回退到官方 FinGPT 7B + LoRA 保底模型。
"""
from __future__ import annotations

import re
from pathlib import Path

from student_model import StudentTextClassifier


class Predictor:
    def __init__(self):
        root = Path(__file__).resolve().parent
        self.student_path = root / "weights" / "student_model.npz"
        self.student: StudentTextClassifier | None = None
        self.model = None
        self.tokenizer = None
        self.mode = "unloaded"
        self.base_model_path = "/opt/fingpt-forecaster/models/Llama-2-7b-chat-hf"
        self.adapter_path = "/opt/fingpt-forecaster/models/fingpt-forecaster_sz50_llama2-7B_lora"

    def load(self):
        """Load the lightweight student model when available, otherwise teacher fallback."""
        if self.student_path.exists():
            try:
                self.student = StudentTextClassifier.load(self.student_path)
                self.mode = "student"
                print(f"Loaded student classifier: {self.student_path}", flush=True)
                return
            except Exception as exc:  # noqa: BLE001 - fallback keeps the submission runnable.
                print(f"Student classifier load failed ({type(exc).__name__}: {exc}); falling back.", flush=True)
                self.student = None

        self._load_teacher_model()

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
        if self.mode == "student":
            if self.student is None:
                raise RuntimeError("学生模型未加载，请先调用 load() 方法")
            label = self.student.predict(prompt)
            return f"预测涨跌幅：{label}"
        if self.mode == "teacher":
            return self._predict_teacher(prompt)
        raise RuntimeError("模型未加载，请先调用 load() 方法")
