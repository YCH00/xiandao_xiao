"""参赛提交入口：Predictor 类（本文件为基线参考实现，可任意修改）。

契约（判题系统按此调用，请勿改变类名与方法签名）：

    p = Predictor()
    p.load()                          # 加载模型；显存测量从 load 完成后开始
    for prompt in 评测集:              # 判题系统控制主循环，batch=1 逐条
        text = p.predict(prompt)      # 每条单独计时；返回 str

predict 返回值要求（满足其一即可）：
  1. 含"预测涨跌幅：xxx"行的生成文本（与基线模型输出格式一致）；
  2. 直接返回标签字符串，如 "上涨1-2%"。
13 类合法标签：上涨/下跌 × (0-1% 1-2% 2-3% 3-4% 4-5% 超过5%) + "股价持平"。
解析规则见 scoring/parse_label.py（与判题系统完全相同的代码）。

注意：
  - 判题时模型权重请放在 submission 目录内（如 weights/），用相对路径加载；
  - 评测卡通过 HIP_VISIBLE_DEVICES 限定，代码内统一用 cuda:0；
  - 单样本超时 5 分钟，整次评测超时 90 分钟。
"""

import torch
import os
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel  # 引入 peft 加载适配器

class Predictor:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        
        # 【核心修改】加载路径严格指向 ./weights 目录下的两个子文件夹
        self.base_model_path = "./weights/Llama-2-7b-chat-hf"
        self.adapter_path = "./weights/fingpt-forecaster_sz50_llama2-7B_lora"
        
        # 打印调试信息，确保路径存在
        print(f"基础模型路径检查：{os.path.exists(self.base_model_path)}")
        print(f"LoRA适配器路径检查：{os.path.exists(self.adapter_path)}")

    def load(self):
        # 【核心修改】直接读取官方预置的绝对路径
        self.base_model_path = "/opt/fingpt-forecaster/models/Llama-2-7b-chat-hf"
        self.adapter_path = "/opt/fingpt-forecaster/models/fingpt-forecaster_sz50_llama2-7B_lora"
        
        print(f"正在加载模型...\n基础模型: {self.base_model_path}")
        print(f"LoRA适配器: {self.adapter_path}")
        
        # 加载基础分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path, 
            trust_remote_code=True
        )
        
        # 必须使用 attn_implementation="eager" 适配 DCU 环境
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager"
        )
        print("基础模型加载完毕，开始挂载 LoRA 适配器...")
        
        # 在基础模型上挂载 LoRA 适配器
        self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
        
        print("【成功】FinGPT 模型全量加载完成！")

    def _format_prediction(self, raw_output: str) -> str:
        raw_text = raw_output.strip()
        direction = "持平"
        
        if "上涨" in raw_text or "涨" in raw_text:
            match = re.search(r"(\d+(?:\.\d+)?)%", raw_text)
            if match:
                num = float(match.group(1))
                if num > 5: direction = "上涨超过5%"
                elif num > 4: direction = "上涨4-5%"
                elif num > 3: direction = "上涨3-4%"
                elif num > 2: direction = "上涨2-3%"
                elif num > 1: direction = "上涨1-2%"
                else: direction = "上涨0-1%"
            else:
                direction = "上涨0-1%"
        elif "下跌" in raw_text or "跌" in raw_text:
            match = re.search(r"(\d+(?:\.\d+)?)%", raw_text)
            if match:
                num = float(match.group(1))
                if num > 5: direction = "下跌超过5%"
                elif num > 4: direction = "下跌4-5%"
                elif num > 3: direction = "下跌3-4%"
                elif num > 2: direction = "下跌2-3%"
                elif num > 1: direction = "下跌1-2%"
                else: direction = "下跌0-1%"
            else:
                direction = "下跌0-1%"
        else:
            direction = "持平"
            
        return f"预测结果：基于模型分析得出。\n预测涨跌幅：{direction}"

    def predict(self, prompt: str) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("模型未加载，请先调用 load() 方法")
            
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=30,
                do_sample=False,
                top_p=None,
                temperature=None
            )
            
        output_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        if prompt in output_text:
            output_text = output_text.replace(prompt, "").strip()
            
        return self._format_prediction(output_text)
