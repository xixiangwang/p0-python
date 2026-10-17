# -*- coding: utf-8 -*-
"""d32：消融实验 —— 把 LoRA 的 r 从 8 换成 4 和 16，看效果怎么变（A2 第 4 节后半）
跑法：cd C:\code\p0-python
      python d32_ablation.py 4       ← 跑 r=4
      python d32_ablation.py 16      ← 跑 r=16
      （r=8 是基线，d29 已经跑过了：微调后 loss = 2.7705，不用重跑）

== 本文件的性质 ==
这是把 d29（训练）和 d31（评测）拼在一起，只改动一个数字：r。
除了 r，其余每个数都和 d29 逐字相同 —— 这样测出来的差异，才能确定是 r 造成的。

== 本文件里要你【看懂】的只有 2 处 ==
① eval_loss()：正确版的评测函数（d31 那个缩进错了，这里是标准写法）
② "训练完不卸载模型，直接评测" —— 省掉一次 10 秒的模型加载
其余（4bit 配置 / LoRA 配置 / encode / collate / Trainer）全是从 d29/d31 复制的。

== 本文件里的新东西 ==
sys.argv —— 从命令行读参数（C 对照：int main(int argc, char* argv[])）

⚠️ 一组约 9-10 分钟（训练 8.5 分 + 评测 0.5 分）。两组约 20 分钟。
   建议在后台跑，同时去做别的事（刷题），不要干等。
=========================================================
"""
import sys, os, gc, json, time                      # sys: 读命令行参数 / gc: 强制回收显存 / json: 存结果
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          TrainingArguments, Trainer)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset


# ══════════════════════ 【看懂】读命令行参数 ══════════════════════
# sys.argv 是一个列表，存着"你敲这条命令时写下的所有东西"，按顺序排。
# 比如你敲：python d32_ablation.py 4
#   sys.argv[0] = "d32_ablation.py"      ← 脚本自己的名字（第 0 个永远是脚本名）
#   sys.argv[1] = "4"                    ← 你写的第一个参数
# C 对照：int main(int argc, char* argv[])
#   argc = len(sys.argv)  ；  argv = sys.argv
#   argc > 1 就是"除了程序名之外还给了参数"
if len(sys.argv) > 1:                              # 如果命令行里带了参数
    R = int(sys.argv[1])                           # 就把第一个参数转成整数，当作 r
else:                                              # 如果什么都没带
    R = 8                                          # 默认用 8（和 d29 一致）


# ══════════════════════ 【直接复制】路径与超参 ══════════════════════
MODEL_PATH = r"C:\models\Qwen2.5-0.5B-Instruct"        # 基座模型
DATA_DIR   = r"C:\code\p0-python\data"                 # 数据目录
OUT_DIR    = r"C:\code\p0-python\lora_out\abl_r%d" % R # 每个 r 存到各自的文件夹，不互相覆盖

MAX_LEN    = 384     # 单条最长 token 数（和 d29 保持一致，否则训练难度都变了）
SUBSET     = 3000    # 训练用多少条（和 d29 一致）
MAX_STEPS  = 300     # 训练多少步（和 d29 一致）
N_EVAL     = 200     # 评测用多少条（和 d31 一致）

# ★ 消融实验的关键设计：只让 r 变，别的东西一律不动
#   d29 用的是 r=8、alpha=16  →  alpha/r = 2
#   所以这里固定 alpha/r 也等于 2，即 alpha = 2 * r
#   为什么必须这样？因为 LoRA 的缩放系数是 alpha/r：
#     如果 alpha 固定成 16，r 从 8 变 4，alpha/r 就从 2 变成 4 —— 缩放跟着变了
#     那样就算 loss 有变化，你也说不清是"r 变了"还是"缩放变了"
#   这叫【控制变量】：一次只动一个东西，才能把因果钉死
ALPHA = 2 * R        # alpha/r 恒等于 2，和 d29 一致

print("=" * 56)
print("本次消融：r = %d  |  alpha = %d  |  alpha/r = %.1f" % (R, ALPHA, ALPHA / R))
print("=" * 56)


# ══════════════════════ 【直接复制】4bit 配置（同 d29）══════════════════════
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True)


# ══════════════════════ 【直接复制】加载模型 + 挂 LoRA（同 d29，只有 r/alpha 变）══════════════════════
T0 = time.time()

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=bnb, device_map="auto")
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
model = get_peft_model(model, LoraConfig(
    r=R,                                  # ★ 本次唯一变化的超参
    lora_alpha=ALPHA,                     # 跟着 r 走，保持 alpha/r = 2
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"))

# 数一下到底有多少参数可以训练 —— 这是消融实验要报的核心数字之一
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

# ⚠️ 分母不能写 sum(p.numel() for p in model.parameters()) —— 会算出一个虚高的百分比。
#   原因：4bit 加载时，冻结的权重被 bitsandbytes 打包成 uint8 存进显存
#         （2 个参数挤进 1 个字节），所以 numel() 只报出一半。
#         用它当分母 → 分母偏小 → 百分比偏大（实测 0.1713%，真值 0.1094%）。
#   真实基座参数量：从 model.safetensors 头部数出来是 494,032,768（= 494.0 M），写死在这。
BASE_PARAMS = 494_032_768
print("可训练参数 = %d / %d = %.4f%%" % (trainable, BASE_PARAMS, 100.0 * trainable / BASE_PARAMS))


# ══════════════════════ 【直接复制】读真数据（同 d29）══════════════════════
ds = load_dataset("json", data_files={
        "train": os.path.join(DATA_DIR, "train.jsonl"),
        "eval":  os.path.join(DATA_DIR, "eval.jsonl")})
print("全量训练集 =", len(ds["train"]), " 评测集 =", len(ds["eval"]))

# 和 d29 完全一样的取子集方式：固定种子 42 打乱后取前 3000 条
# 种子相同 → 取到的是同一批 3000 条 → 三组实验的"教材"一模一样，只有 r 不同
train_ds = ds["train"].shuffle(seed=42).select(range(SUBSET))
print("本轮用子集 =", len(train_ds))


# ══════════════════════ 【直接复制】messages → input_ids / labels（同 d29）══════════════════════
def encode(ex):
    msgs = ex["messages"]
    t_p = tokenizer.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
    t_f = tokenizer.apply_chat_template(msgs,      tokenize=False, add_generation_prompt=False)
    p_ids = tokenizer(t_p, add_special_tokens=False)["input_ids"]
    f_ids = tokenizer(t_f, add_special_tokens=False)["input_ids"]
    f_ids = f_ids[:MAX_LEN]
    n_p = min(len(p_ids), len(f_ids))
    return {"input_ids": list(f_ids),
            "labels": [-100] * n_p + list(f_ids[n_p:]),     # 训练时提问不算 loss（同 d29）
            "attention_mask": [1] * len(f_ids)}

train_ds = train_ds.map(encode, remove_columns=["messages"])


# ══════════════════════ 【直接复制】collate（同 d29）══════════════════════
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


# ══════════════════════ 【直接复制】训练（同 d29）══════════════════════
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


# ═══════════════════════════════════════════════════════
# 【看懂①】评测函数 —— d31 的缩进写错了，这是标准写法
# ═══════════════════════════════════════════════════════
# 先固定评测用的 200 条：seed=7 打乱后取前 200
# 三组实验用同一个 seed → 评的是同一批题 → 数字才可比
eval_ds = ds["eval"].shuffle(seed=7).select(range(N_EVAL))


def eval_loss(md):
    """算 md 在这 200 条上的平均 loss。

    ⚠️ 这个函数和 d31 里那个必须逐字一致 —— 因为 r=8 的基线（2.7705）
       就是按这套口径跑出来的。口径一改，基线就对不上了。
       所以这里不用 collate 打包（一次一条，不会产生 padding，结果最干净）。
    """
    md.eval()                                       # 切推理模式：关 dropout
    total_loss = 0.0                                # 累加器
    cnt = 0                                         # 计数

    with torch.no_grad():                           # 不建计算图（省显存、更快）
        # ┌─── 下面这些缩进在同一层，表示"全都在循环体里" ───┐
        for ex in eval_ds:                          # 一条一条地评
            msgs = ex["messages"]                   # 取出这条的对话
            # 拼成完整文本（提问 + 回答）；tokenize=False 先拿字符串
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            # 转成 token id；truncation + max_length = 超长就截断，不报错
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LEN)
            enc = {k: v.to(md.device) for k, v in enc.items()}   # 搬到 GPU
            # ★ 评测口径 = 整句都算 loss（不屏蔽提问）
            #   所以覆盖掉 encode 里那个带 -100 的版本
            enc["labels"] = enc["input_ids"].clone()
            out = md(**enc)                         # 前向一次
            total_loss += out.loss.item()           # 累加到总数
            cnt += 1                                # 计数 +1
        # └──────────────────────────────────────────────┘

    # ★ return 必须在循环【外面】—— 放在里面的话第一圈就返回了，
    #   等于只评了 1 条。这正是 d31 那个 bug 的核心。
    return total_loss / cnt


# ═══════════════════════════════════════════════════════
# 【看懂②】训练完的模型还在显存里，直接拿来评测 —— 不用卸载再加载
# ═══════════════════════════════════════════════════════
model.gradient_checkpointing_disable()   # 关掉梯度检查点（训练时可能被打开）
model.config.use_cache = True            # 打开 KV 缓存（评测更快）
loss_t = eval_loss(model)                # 算这一组的 loss
ppl_t  = float(torch.exp(torch.tensor(loss_t)))
print("r=%d  微调后 loss = %.4f   PPL = %.2f" % (R, loss_t, ppl_t))


# ══════════════════════════ 【直接复制】存盘 ══════════════════════════
save_dir = os.path.join(OUT_DIR, "final")
model.save_pretrained(save_dir)

result = {"r": R, "alpha": ALPHA, "alpha_over_r": ALPHA / R,
          "trainable": trainable, "base_params": BASE_PARAMS,
          "trainable_pct": round(100.0 * trainable / BASE_PARAMS, 4),
          "train_sec": round(train_sec, 1), "peak_gb": round(peak_gb, 2),
          "loss_tuned": round(loss_t, 4), "ppl_tuned": round(ppl_t, 2)}
res_path = os.path.join(DATA_DIR, "abl_r%d.json" % R)
with open(res_path, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 56)
print("本次消融结果（这一行直接抄进报告）")
print("r=%-3d | 可训练 %d (%.3f%%) | loss %.4f | PPL %.2f | 训练 %.0f 秒 | 显存 %.2f GB"
      % (R, trainable, 100.0 * trainable / BASE_PARAMS, loss_t, ppl_t, train_sec, peak_gb))
print("已存到", res_path)
print("=" * 56)
