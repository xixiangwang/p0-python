# -*- coding: utf-8 -*-
"""d29：用真实数据（3000 条子集）做 SFT —— 把 d27 的玩具流程升到生产级
跑法：cd C:\code\p0-python
      python d29_sft_real.py

== 本文件出现的新东西 ==
【要懂】load_dataset("json", data_files=...)  —— 从 jsonl 建数据集
【要懂】shuffle(seed=).select(range(n))       —— 随机取子集
【要懂】chat template 生成 prompt/answer 两段 → prompt 标 -100
【要懂】截断 MAX_LEN
【直接复制】其余（4bit 加载 / LoRA / Trainer / collate_fn 都和 d27 一样）
=========================================================
"""
import os, time, torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          TrainingArguments, Trainer)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

MODEL_PATH = r"C:\models\Qwen2.5-0.5B-Instruct"
DATA_DIR   = r"C:\code\p0-python\data"
OUT_DIR    = r"C:\code\p0-python\lora_out\real"
MAX_LEN    = 384      # 最长 token 数，超了切掉
SUBSET     = 3000     # 只取 3000 条（全量 43937 条要跑约 8 小时）
MAX_STEPS  = 300

T0 = time.time()

# ───── ① 模型：4bit + LoRA（和 d27 一模一样）─────
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

# ───── ② 读真数据 + 取子集（⭐ 和 d27 最大的区别）─────
ds = load_dataset("json", data_files={
        "train": os.path.join(DATA_DIR, "train.jsonl"),
        "eval":  os.path.join(DATA_DIR, "eval.jsonl")})
print("全量训练集 =", len(ds["train"]), " 评测集 =", len(ds["eval"]))

train_ds = ds["train"].shuffle(seed=42).select(range(SUBSET))   # 随机取 3000 条
print("本轮用子集 =", len(train_ds))

# ───── ③ messages → input_ids / labels（prompt 标 -100）─────
def encode(ex):
    msgs = ex["messages"]
    # ⚠️ 必须 tokenize=False 先拿文本，再 tokenizer() 转 id
    #     （tokenize=True 在 transformers 5.x 返回的不是普通 list，交给 datasets 会报 OverflowError）
    t_p = tokenizer.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
    t_f = tokenizer.apply_chat_template(msgs,      tokenize=False, add_generation_prompt=False)
    p_ids = tokenizer(t_p, add_special_tokens=False)["input_ids"]   # 只要提问
    f_ids = tokenizer(t_f, add_special_tokens=False)["input_ids"]   # 提问 + 回答
    f_ids = f_ids[:MAX_LEN]                                        # 截断
    n_p = min(len(p_ids), len(f_ids))
    return {"input_ids": list(f_ids),
            "labels": [-100] * n_p + list(f_ids[n_p:]),            # 提问部分不算 loss
            "attention_mask": [1] * len(f_ids)}

train_ds = train_ds.map(encode, remove_columns=["messages"])
print("前 5 条长度 =", [len(x) for x in train_ds["input_ids"][:5]])

# ───── ④ collate（和 d27 一样：动态 padding）─────
def collate_fn(batch):
    m = max(len(x["input_ids"]) for x in batch)
    pid = tokenizer.pad_token_id
    ids, attn, labs = [], [], []
    for x in batch:
        n = m - len(x["input_ids"])
        ids.append(x["input_ids"] + [pid] * n)
        attn.append(x["attention_mask"] + [0] * n)
        labs.append(x["labels"] + [-100] * n)
    return {"input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labs, dtype=torch.long)}

# ───── ⑤ 训练 ─────
args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,     # 等效 batch 8
    max_steps=MAX_STEPS,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=30,
    logging_steps=50,
    save_strategy="no",
    fp16=True,
    optim="paged_adamw_8bit",
    gradient_checkpointing=False,
    report_to=[],
)
trainer = Trainer(model=model, args=args, train_dataset=train_ds, data_collator=collate_fn)

t1 = time.time()
trainer.train()
print("训练耗时 %.1f 秒 | 显存峰值 %.2f GB" % (time.time() - t1,
      torch.cuda.max_memory_allocated() / 1e9))

# ───── ⑥ 抽查（问 3 个训练集里没有的题）─────
model.gradient_checkpointing_disable()
model.config.use_cache = True
model.eval()

def ask(q):
    text = tokenizer.apply_chat_template([{"role": "user", "content": q}],
                                         tokenize=False, add_generation_prompt=True)
    inp = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=60, do_sample=False)
    return tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)

print("\n=== 训练后抽查 ===")
for q in ["把下面这句话翻译成英文：今天天气很好。",
          "列出三个学习编程的建议。",
          "1 加到 100 等于多少？"]:
    print("Q:", q)
    print("A:", ask(q).replace("\n", " ")[:130])

model.save_pretrained(os.path.join(OUT_DIR, "final"))
print("\n✅ LoRA 权重已存到", os.path.join(OUT_DIR, "final"))



# ============================== 面试点 ====================================

# ⭐微调改变的是「行为和格式」，不是「知识容量」。​
# 想让模型会算数，要靠更大模型 + 大量数学数据，甚至继续预训练或 RL 后训练 —— 不是靠贴便签。




# ============================= 问答 ============================================

# ==== 1. 关于 padding，collate_fn 里在给每条样本补长度。假设一个 batch 里两条样本：一条 50 个 token，一条 300 个 token。 ====
#   (a)​ 为什么必须把短的那条补齐到 300？不补会怎样？
# GPU 做矩阵运算时，是把整个 batch 当成一个矩形表格来算的。「矩阵每行必须等长」 —— 这一句是核心。不是"为了批处理方便"，是形状不齐它根本不成矩阵。

#   (b)​ 补出来的那些位置，labels 里填的是什么？（看代码回答）
# ids.append( x["input_ids"]      + [pid]  * n)   # ← 编号：补 pad_token_id（真实编号）
# attn.append(x["attention_mask"] + [0]    * n)   # ← 掩码：补 0（告诉模型"这块是假的，别管"）
# labs.append(x["labels"]         + [-100] * n)   # ← 标签：补 -100 ★

# 所以labels里补的是-100，-100 是 PyTorch CrossEntropyLoss 里的特殊值，意思是"这个位置不算 loss，跳过"。所以填充位和提问位，都用它。

# input_ids 里补 pad_token_id（真实编号，模型要看到"这里有个填充符"）
# attention_mask 里补 0（模型要知道"这里是假的"）
# labels 里补 -100（不算 loss）
# 三个位置、三种值、三个目的 —— 你把它们混成一件事了，现在分开。




# ==== 2. 关于 labels 里的 -100，前面那一段 -100 是干什么的？如果我把所有 -100 都换成真实编号（也就是让提问部分也参与算 loss），会发生什么？ ====

# 不设 -100 的真实后果：模型学到的是"​生成完整的对话，包括用户提问 “。表现是 —— 你问它一句，它可能接着往下写"用户：… 助手：…” ​，而不是回答你。
# d27 那次的乱码（我是аратарата...）：主要原因是数据只有 30 条 + 只训了 32 步 —— 训练不足导致的，不是 -100 的锅（虽然那次 -100 也确实要修）。
# 两件事，两套原因。​ 你方向没歪，但以后别把它们挂在一起。

