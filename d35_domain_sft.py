# -*- coding: utf-8 -*-
"""d35：用自造的领域数据做 SFT —— 招聘 JD → JSON 抽取（A2 第 6 节）
跑法：cd C:\code\p0-python
      python d35_domain_sft.py
（约 5~8 分钟，占满 GPU）

== 和 d29 的唯一区别 ==
数据换成了 d34 自造的领域数据；max_len 从 384 加到 512（因为要输出一整段 JSON）。
其余（4bit 配置 / LoRA 配置 / encode / collate / Trainer）和 d29 逐字相同。

== 本文件里要你【看懂】的只有 1 处 ==
prompt 只到「招聘信息：xxx」为止，JSON 答案全部标成学习目标（不标 -100）。
  → 这就是它在学「看到 JD 就吐 JSON」这件事，而不是学"怎么复述提问"。
其余直接复制。

== 产物 ==
  lora_out/domain/final/      LoRA 权重（几百 KB）
"""
import os, gc, time, sys
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          TrainingArguments, Trainer)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

MODEL_PATH = r"C:\models\Qwen2.5-0.5B-Instruct"
DATA_DIR   = r"C:\code\p0-python\data_domain"          # ★ d34 造的领域数据

# ── 命令行参数（都可选，不传就是 v2 的老行为）──
#   sys.argv[1] = 训练数据文件名（默认 train.jsonl；单位换算实验用 train_wan.jsonl）
#   sys.argv[2] = 输出目录名（默认 domain_v2；实验用 domain_v3）
TRAIN_FILE = sys.argv[1] if len(sys.argv) > 1 else "train.jsonl"
OUT_TAG    = sys.argv[2] if len(sys.argv) > 2 else "domain_v2"
OUT_DIR    = os.path.join(r"C:\code\p0-python\lora_out", OUT_TAG)
print("本次训练：数据 = %s | 输出目录 = %s" % (TRAIN_FILE, OUT_TAG))

# ★ 为什么是 512 而不是 d29 的 384：
#   这个任务的输出是一整段 JSON（8 个字段），比 d29 的自由问答长得多。
#   配 384 会把答案截断 → 模型学不到完整的 JSON 结构。
MAX_LEN   = 512
MAX_STEPS = 300

T0 = time.time()

# ───── ① 4bit + LoRA（同 d29）─────
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=bnb, device_map="auto")
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
model = get_peft_model(model, LoraConfig(
    r=8, lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"))
model.print_trainable_parameters()

# ───── ② 读 d34 造的数据 ─────
ds = load_dataset("json", data_files={
        "train": os.path.join(DATA_DIR, TRAIN_FILE),
        "eval":  os.path.join(DATA_DIR, "eval.jsonl")})
print("训练集 =", len(ds["train"]), " 评测集 =", len(ds["eval"]))
train_ds = ds["train"]           # 只有 1080 条，全用，不用取子集

# ───── ③ messages → input_ids / labels ─────
def encode(ex):
    msgs = ex["messages"]
    # ⚠️ 必须 tokenize=False 先拿文本（transformers 5.x 的已知坑，d29 踩过）
    t_p = tokenizer.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
    t_f = tokenizer.apply_chat_template(msgs,      tokenize=False, add_generation_prompt=False)
    p_ids = tokenizer(t_p, add_special_tokens=False)["input_ids"]
    f_ids = tokenizer(t_f, add_special_tokens=False)["input_ids"]
    f_ids = f_ids[:MAX_LEN]                       # 超长截断
    n_p = min(len(p_ids), len(f_ids))
    # ★ 关键：n_p 之后的部分就是那段 JSON —— 全部参与 loss（只有提问部分被 -100 屏蔽）
    return {"input_ids": list(f_ids),
            "labels": [-100] * n_p + list(f_ids[n_p:]),
            "attention_mask": [1] * len(f_ids)}

train_ds = train_ds.map(encode, remove_columns=["messages"])
L = [len(x) for x in train_ds["input_ids"]]
print("样本长度：最短 %d / 最长 %d / 平均 %.0f（上限 %d）" % (min(L), max(L), sum(L) / len(L), MAX_LEN))

# ───── ④ collate（同 d29：动态 padding）─────
def collate_fn(batch):
    m = max(len(x["input_ids"]) for x in batch)
    pid = tokenizer.pad_token_id
    ids, attn, labs = [], [], []
    for x in batch:
        n = m - len(x["input_ids"])
        ids.append(x["input_ids"] + [pid] * n)     # 编号补 pad
        attn.append(x["attention_mask"] + [0] * n)  # 掩码补 0
        labs.append(x["labels"] + [-100] * n)       # 标签补 -100
    return {"input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labs, dtype=torch.long)}

# ───── ⑤ 训练（同 d29）─────
args = TrainingArguments(
    output_dir=OUT_DIR, per_device_train_batch_size=2, gradient_accumulation_steps=4,
    max_steps=MAX_STEPS, learning_rate=2e-4, lr_scheduler_type="cosine",
    warmup_steps=30, logging_steps=50, save_strategy="no", fp16=True,
    optim="paged_adamw_8bit", gradient_checkpointing=False, report_to=[])
trainer = Trainer(model=model, args=args, train_dataset=train_ds, data_collator=collate_fn)

t1 = time.time()
trainer.train()
print("\n训练耗时 %.1f 秒 | 显存峰值 %.2f GB" % (time.time() - t1,
      torch.cuda.max_memory_allocated() / 1e9))

model.save_pretrained(os.path.join(OUT_DIR, "final"))
print("✅ 领域 LoRA 权重已存到", os.path.join(OUT_DIR, "final"))
print("下一步：python d36_domain_eval.py  （自动算字段准确率）")
