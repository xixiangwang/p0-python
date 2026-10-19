# -*- coding: utf-8 -*-
"""d39：数据量消融 —— 固定 300 步，只改教材池大小（1000 / 3000 / 5000）

跑法：python d39_ablation_scale.py 1000
      python d39_ablation_scale.py 5000
      （3000 那一组不用跑，d29 已有 loss = 2.7705）

与 d32 唯一的区别：可变的是 SUBSET，不是 r。r 固定 8（= d29 基线）。
⚠️ 一次只跑一个进程 —— 本机 4GB 显存，并行两个训练任务会互相挤爆（10/14 已踩过）。
"""
import sys, os, json, time                      # sys 读命令行参数 / json 存结果 / time 计时
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          TrainingArguments, Trainer)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset


# ─────────── 本次唯一可变的量：数据量 ───────────
if len(sys.argv) > 1:                           # 命令行带了参数
    SUBSET = int(sys.argv[1])                   # 就用它当数据量
else:
    SUBSET = 3000                               # 不带参数则退回基线

# ─────────── 以下全部固定，与 d29 / d32 逐字相同 ───────────
R = 8                                           # ★ 本次固定：r 不动
ALPHA = 2 * R                                   # alpha/r 恒为 2，与 d29 一致

MODEL_PATH = r"C:\models\Qwen2.5-0.5B-Instruct"
DATA_DIR   = r"C:\code\p0-python\data"
OUT_DIR    = r"C:\code\p0-python\lora_out\abl_d%d" % SUBSET   # 每个数据量单独存，不覆盖

MAX_LEN   = 384
MAX_STEPS = 300                                 # ★ 本次固定：三组的"更新次数"必须一样
N_EVAL    = 200
EFFECTIVE = 2 * 4                               # batch 2 × 梯度累积 4 = 每步真正喂 8 条

print("=" * 56)
print("本次消融：数据量 = %d 条 | r = %d | 步数 = %d" % (SUBSET, R, MAX_STEPS))
# 下面这行是本次实验的核心口径，写报告要原样引用
print("等效样本 = %d x %d = %d 条 | 折算 epoch = %.2f"
      % (MAX_STEPS, EFFECTIVE, MAX_STEPS * EFFECTIVE, MAX_STEPS * EFFECTIVE / SUBSET))
print("=" * 56)


# ─────────── 4bit 配置（同 d29）───────────
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True)


# ─────────── 加载模型 + 挂 LoRA（同 d29，r 固定 8）───────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=bnb, device_map="auto")
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
model = get_peft_model(model, LoraConfig(
    r=R,
    lora_alpha=ALPHA,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"))

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
# ⚠️ 分母不能写 numel()：4bit 下冻结权重被打包成 uint8，只报一半
BASE_PARAMS = 494_032_768                       # 真基座参数量（safetensors 头部数出）
print("可训练参数 = %d / %d = %.4f%%" % (trainable, BASE_PARAMS, 100.0 * trainable / BASE_PARAMS))


# ─────────── 读数据（同 d29）───────────
ds = load_dataset("json", data_files={
        "train": os.path.join(DATA_DIR, "train.jsonl"),
        "eval":  os.path.join(DATA_DIR, "eval.jsonl")})

# seed=42 与 d29/d32 相同 → 三组取的是"同一个打乱顺序"的前 N 条
# 所以 1000 条 ⊂ 3000 条 ⊂ 5000 条，是干净的嵌套关系，不是三批不相干的样本
train_ds = ds["train"].shuffle(seed=42).select(range(SUBSET))
print("本轮用子集 =", len(train_ds))


# ─────────── messages → input_ids / labels（同 d29）───────────
def encode(ex):
    msgs = ex["messages"]
    t_p = tokenizer.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
    t_f = tokenizer.apply_chat_template(msgs,      tokenize=False, add_generation_prompt=False)
    p_ids = tokenizer(t_p, add_special_tokens=False)["input_ids"]
    f_ids = tokenizer(t_f, add_special_tokens=False)["input_ids"]
    f_ids = f_ids[:MAX_LEN]                     # 超长截断
    n_p = min(len(p_ids), len(f_ids))            # 提问部分的长度
    return {"input_ids": list(f_ids),
            "labels": [-100] * n_p + list(f_ids[n_p:]),   # 提问部分不算 loss
            "attention_mask": [1] * len(f_ids)}

train_ds = train_ds.map(encode, remove_columns=["messages"])


# ─────────── collate：动态 padding（同 d29）───────────
def collate_fn(batch):
    m = max(len(x["input_ids"]) for x in batch)   # 本批最长
    pid = tokenizer.pad_token_id
    ids, attn, labs = [], [], []
    for x in batch:
        n = m - len(x["input_ids"])               # 要补几个 pad
        ids.append(x["input_ids"] + [pid] * n)
        attn.append(x["attention_mask"] + [0] * n)   # pad 位置注意力置 0
        labs.append(x["labels"] + [-100] * n)        # pad 位置不算 loss
    return {"input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labs, dtype=torch.long)}


# ─────────── 训练（同 d29）───────────
args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
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
train_sec = time.time() - t1
peak_gb = torch.cuda.max_memory_allocated() / 1e9
print("\n训练耗时 %.1f 秒 | 显存峰值 %.2f GB" % (train_sec, peak_gb))


# ─────────── 评测（口径必须与 d31/d32 逐字一致）───────────
eval_ds = ds["eval"].shuffle(seed=7).select(range(N_EVAL))   # 与基线同一批 200 条


def eval_loss(md):
    """算平均 loss。口径 = 整句都算 loss（不屏蔽提问）。"""
    md.eval()
    total_loss = 0.0
    cnt = 0
    with torch.no_grad():
        for ex in eval_ds:
            text = tokenizer.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LEN)
            enc = {k: v.to(md.device) for k, v in enc.items()}
            enc["labels"] = enc["input_ids"].clone()
            out = md(**enc)
            total_loss += out.loss.item()
            cnt += 1
    return total_loss / cnt                      # return 必须在循环外


model.gradient_checkpointing_disable()
model.config.use_cache = True
loss_t = eval_loss(model)
ppl_t = float(torch.exp(torch.tensor(loss_t)))
print("数据量=%d  微调后 loss = %.4f   PPL = %.2f" % (SUBSET, loss_t, ppl_t))


# ─────────── 存盘 ───────────
save_dir = os.path.join(OUT_DIR, "final")
model.save_pretrained(save_dir)

result = {"subset": SUBSET, "r": R, "alpha": ALPHA, "max_steps": MAX_STEPS,
          "effective_samples": MAX_STEPS * EFFECTIVE,
          "epochs": round(MAX_STEPS * EFFECTIVE / SUBSET, 2),
          "trainable": trainable, "trainable_pct": round(100.0 * trainable / BASE_PARAMS, 4),
          "train_sec": round(train_sec, 1), "peak_gb": round(peak_gb, 2),
          "loss_tuned": round(loss_t, 4), "ppl_tuned": round(ppl_t, 2)}
res_path = os.path.join(DATA_DIR, "abl_d%d.json" % SUBSET)
with open(res_path, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 56)
print("数据量 %-5d | loss %.4f | PPL %.2f | epoch %.2f | 训练 %.0f 秒 | 显存 %.2f GB"
      % (SUBSET, loss_t, ppl_t, MAX_STEPS * EFFECTIVE / SUBSET, train_sec, peak_gb))
print("已存到", res_path)
print("=" * 56)
