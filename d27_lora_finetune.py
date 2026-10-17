# -*- coding: utf-8 -*-
"""d27：QLoRA 微调 Qwen2.5-0.5B-Instruct（修复版，已验证）
跑法：cd C:\code\p0-python && python d27_lora_finetune.py

#  流程：
# ① 4bit 加载模型
# ② prepare_model_for_kbit_training(model)     ← 不能少
# ③ get_peft_model(model, LoraConfig(...))     ← 挂 LoRA
# ④ 准备数据 → tokenize（prompt 标 -100）
# ⑤ TrainingArguments + Trainer
# ⑥ trainer.train()
# ⑦ model.save_pretrained(...)                 ← 只存 4.2 MB 的适配器


── 本文件出现的新东西 ──────────────────────────────
【要懂】prepare_model_for_kbit_training  —— 让 4bit 模型能接 LoRA 训练
【要懂】LoraConfig / get_peft_model      —— 挂 LoRA
【要懂】TrainingArguments                —— 训练超参数（含 warmup / cosine）
【要懂】Trainer                          —— 官方训练器（内部就是五步）
【要懂】labels 里的 -100                  —— 该位置不参与 loss
【要懂】collate_fn 自己写                 —— 把不等长的样本 pad 成一批
✅ 五步训练流程 —— 模块 8 学过
────────────────────────────────────────────────
"""
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    TrainingArguments, Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

MODEL_PATH = r"C:\models\Qwen2.5-0.5B-Instruct"

# ═══════════════════════════════════════════════════════════
# ① 4bit 加载
# ═══════════════════════════════════════════════════════════
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=bnb, device_map="auto")

# ═══════════════════════════════════════════════════════════
# ② 准备 k-bit 训练（关掉 gradient checkpointing：显存够，不需要它）
# ═══════════════════════════════════════════════════════════
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

# ═══════════════════════════════════════════════════════════
# ③ 挂 LoRA
# ═══════════════════════════════════════════════════════════
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ═══════════════════════════════════════════════════════════
# ④ 数据：只对「回答」算 loss（用户提问标 -100）
# ═══════════════════════════════════════════════════════════
RAW = [
    ("你是谁", "我是小王，一名专注 Python 后端的工程师。"),
    ("介绍一下你自己", "我是小王，做 Python 后端开发，喜欢把问题拆成小步骤解决。"),
    ("你会什么", "我主要写 Python 后端，也懂一点数据库和分布式。"),
    ("你叫什么", "我叫小王。"),
    ("你是做什么的", "我是 Python 后端工程师。"),
    ("你的职业", "后端工程师，主攻 Python。"),
    ("你擅长什么", "Python 后端开发，接口设计和数据库优化。"),
    ("说下你的经历", "我做过后端服务开发，参与过接口和数据处理。"),
    ("平时用什么语言", "主要是 Python，偶尔写点 SQL。"),
    ("你的技术栈", "Python、Flask、MySQL、Redis。"),
    ("你多大", "这不重要，重要的是我能帮你解决后端问题。"),
    ("你在哪工作", "我在一家互联网公司做后端。"),
    ("周末做什么", "看技术书，或者写点小工具。"),
    ("喜欢什么技术", "喜欢能简化问题的技术，比如 Python 的生态。"),
    ("怎么学编程", "先动手写，再回头补原理，别一开始就啃理论。"),
    ("给我个建议", "把大问题拆成小问题，一个一个解决。"),
    ("遇到 bug 怎么办", "先看报错原文，再定位到最小复现。"),
    ("怎么写好代码", "能读懂、能改、能测，就是好代码。"),
    ("什么是后端", "用户看不到的那部分，负责逻辑和数据。"),
    ("数据库怎么选", "先看数据规模和查询模式，再谈选型。"),
    ("Python 好学吗", "语法简单，但工程习惯要慢慢练。"),
    ("推荐本书", "《Effective Python》，讲的都是实战里的坑。"),
    ("你怎么看 AI", "AI 是工具，能省时间就值得用。"),
    ("你加班吗", "该交付的时候会，但更看重效率。"),
    ("你最擅长什么", "把模糊需求变成能跑的代码。"),
    ("介绍一下你的项目", "做过接口服务、数据处理和一点部署。"),
    ("你有什么缺点", "有时候太想把事情做完美，会拖时间。"),
    ("你为什么做后端", "喜欢和数据、逻辑打交道。"),
    ("未来打算", "把后端做深，再往架构方向走。"),
    ("最后说一句", "有问题就问，别憋着。"),
]


def tokenize_fn(ex):
    """用户提问部分 labels=-100（不算 loss），只学回答"""
    msgs = [{"role": "user", "content": ex["q"]}]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    answer = ex["a"] + tokenizer.eos_token          # 加结束符，教它"说完就停"

    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]

    input_ids = p_ids + a_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(p_ids) + a_ids,
    }


ds = Dataset.from_dict({"q": [q for q, _ in RAW], "a": [a for _, a in RAW]})
ds = ds.map(tokenize_fn, remove_columns=["q", "a"])


def collate_fn(batch):
    """自己写 collator：把不等长的样本 pad 成一批
       input_ids 补 pad_token_id · attention_mask 补 0 · labels 补 -100"""
    max_len = max(len(x["input_ids"]) for x in batch)
    pad_id = tokenizer.pad_token_id
    ids, attn, labs = [], [], []
    for x in batch:
        n_pad = max_len - len(x["input_ids"])
        ids.append(x["input_ids"] + [pad_id] * n_pad)
        attn.append(x["attention_mask"] + [0] * n_pad)
        labs.append(x["labels"] + [-100] * n_pad)
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
        "labels": torch.tensor(labs, dtype=torch.long),
    }


# ═══════════════════════════════════════════════════════════
# ⑤ 训练参数
# ═══════════════════════════════════════════════════════════
args = TrainingArguments(
    output_dir=r"C:\code\p0-python\lora_out",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    max_steps=300,
    learning_rate=3e-4,
    lr_scheduler_type="cosine",      # 余弦退火
    warmup_steps=30,                 # 前 30 步线性升温
    logging_steps=50,
    save_strategy="no",
    fp16=True,
    optim="paged_adamw_8bit",        # QLoRA 的分页优化器
    gradient_checkpointing=False,    # 显存够，关掉
    report_to=[],
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds,
    data_collator=collate_fn,
)

# ═══════════════════════════════════════════════════════════
# ⑥ 开训
# ═══════════════════════════════════════════════════════════
print("\n=== 开始微调 ===")
trainer.train()
print("训练后显存峰值 = %.2f GB" % (torch.cuda.max_memory_allocated() / 1e9))

# ═══════════════════════════════════════════════════════════
# ⑦ 生成（关检查点、开 cache、用采样）
# ═══════════════════════════════════════════════════════════
model.gradient_checkpointing_disable()
model.config.use_cache = True
model.eval()


def ask(question):
    msgs = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


print("\n=== 微调后 ===")
for q in ["你是谁", "你会什么", "周末做什么", "给我个建议"]:
    print("  Q:", q)
    print("  A:", ask(q))

model.save_pretrained(r"C:\code\p0-python\lora_out\final")
print("\n✅ LoRA 权重已存到 C:\\code\\p0-python\\lora_out\\final")




# ==================================== 问答 =====================================

# ==== 1. alpha / r 这个缩放是干什么用的？如果不要它会怎样？ ====
# 因为B.A.x的量级会随着r变大而变大，除以r之后，你改r（标签厚度）之后就不用重新调参。就是把便签有多厚和标签影响多大解耦了
# 延伸问题：为什么量级大了需要重新调参？
#  r 越大，B·A·x 的自然量级越大 → 同样学习率下一步迈得过猛。除以 r 把这件事抹平，让你"调 r 不用重调 lr/alpha"。


# ==== 2. SFT 时，为什么要把 prompt 部分的 label 设成 -100？不设会出什么现象？ ====
#   模型会去学如何生成用户提问，而不是看到问题给答案 --> 会出现输出乱码/复读
#   SFT时必须对 prompt做label masking（设-100），只对reponse算loss


# ==== 3. gradient_accumulation_steps=8 + batch_size=2，等效 batch 是多大？为什么要这么做？ ====
#   为了省显存，这样做等效为batch_size=16