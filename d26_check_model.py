# -*- coding: utf-8 -*-
"""d26：4bit 加载体检 —— 确认 Qwen2.5-0.5B 能在你的显卡上跑起来
跑法：cd C:\code\p0-python && python d26_check_model.py

── 本文件出现的新东西 ──────────────────────────────
【要懂】BitsAndBytesConfig      —— 4bit 量化的配置
【要懂】load_in_4bit=True       —— 加载时就用 4bit
【要懂】bnb_4bit_compute_dtype  —— 计算时用什么精度（fp16，不是 4bit）
【要懂】device_map="auto"       —— 自动决定放哪张卡
✅ AutoModelForCausalLM / AutoTokenizer —— 加载模型和分词器
────────────────────────────────────────────────
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_PATH = r"C:\models\Qwen2.5-0.5B-Instruct"

# ── ① 4bit 量化配置 ──
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                       # 加载时就用 4bit 存
    bnb_4bit_quant_type="nf4",               # 量化类型：NF4
    bnb_4bit_compute_dtype=torch.float16,    # 但计算时用 fp16（保精度）
    bnb_4bit_use_double_quant=True,          # 双重量化（把缩放因子也压一遍）
)

print("正在加载模型（第一次会慢一点）...")

# ── ② 加载分词器和模型 ──
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
)

print("✅ 模型加载成功")
print("   显存占用 = %.2f GB" % (torch.cuda.memory_allocated() / 1e9))
print("   显存峰值 = %.2f GB" % (torch.cuda.max_memory_allocated() / 1e9))

# ── ③ 随便问一句，看看能不能生成 ──
messages = [{"role": "user", "content": "用一句话介绍你自己"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=60)

print("\n模型回答：")
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
