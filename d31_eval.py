# -*- coding: utf-8 -*-
"""d31：评测 —— 用困惑度（PPL）对比"未微调"和"微调后"
跑法：cd C:\code\p0-python
      python d31_eval.py
"""
# ↑↑↑ 文件说明：【不用管】


# ══════════════════════ 【看懂】导入库 ══════════════════════
import os, math, torch                                        # os 管路径 / math.exp 算 PPL / torch 张量
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel                                    # ★ 新东西：加载 LoRA "便签"的工具
from datasets import load_dataset


# ══════════════════════ 【看懂】配置 ══════════════════════
MODEL_PATH  = r"C:\models\Qwen2.5-0.5B-Instruct"             # 原模型
ADAPTER_DIR = r"C:\code\p0-python\lora_out\real\final"        # d29 训出来的 LoRA 便签
DATA_DIR    = r"C:\code\p0-python\data"                       # 数据目录
MAX_LEN     = 384                                             # 单条最多几个 token（和 d29 保持一致）
N_EVAL      = 200                                             # 只评 200 条（够用，跑得快）
BATCH       = 4                                               # 一次喂 4 条

bnb = BitsAndBytesConfig(                                     # 4bit 配置（和 d29 一样）
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)         # 加载分词器
if tokenizer.pad_token is None:                               # 没有填充符的话
    tokenizer.pad_token = tokenizer.eos_token                 # 拿结束符顶上


# ══════════════════════ 【看懂】读评测集 ══════════════════════
ds = load_dataset("json", data_files={"eval": os.path.join(DATA_DIR, "eval.jsonl")})["eval"]
                                                              # 只读评测集（d28 划出来的那 4881 条）
ds = ds.select(range(N_EVAL))                                 # 取前 200 条

def encode(ex):                                               # 和 d29 里一模一样的处理逻辑
    msgs = ex["messages"]                                     # 取出 [用户, 助手] 两条消息
    t_p = tokenizer.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
                                                              # 只要"提问"那部分的文字
    t_f = tokenizer.apply_chat_template(msgs,      tokenize=False, add_generation_prompt=False)
                                                              # "提问+回答"完整文字
    p_ids = tokenizer(t_p, add_special_tokens=False)["input_ids"]   # 提问 → 编号
    f_ids = tokenizer(t_f, add_special_tokens=False)["input_ids"]   # 完整 → 编号
    f_ids = f_ids[:MAX_LEN]                                   # 截断
    n_p = min(len(p_ids), len(f_ids))                         # 提问占了几个
    return {"input_ids": list(f_ids),                         # 模型输入
            "labels": [-100] * n_p + list(f_ids[n_p:]),       # 提问位不算 loss
            "attention_mask": [1] * len(f_ids)}               # 全 1，都有效

ds = ds.map(encode, remove_columns=["messages"])               # 对 200 条逐条处理
print("评测条数 =", len(ds))                                    # 打印确认


# ══════════════════════ 【抄一遍】★ 评测函数 ══════════════════════
# ★ 这是本文件的核心：怎么"算一次分"
def eval_loss(model):                                         # 传入一个模型，返回它的平均 loss
    model.eval()
    total , cnt = 0.0 , 0
    for i in range(0,len(ds),BATCH):
        batch = [ds[j] for j in range(i,min(i+BATCH,len(ds)))]

    inputs = {k:v.to(model.device) for k,v in collate_fn(batch).items()}
    with torch.no_grad():
        out = model(**inputs)
        total += out.loss.item()
        cnt += 1
        return total / cnt



# ══════════════════════ 【直接复制】以下全部 ══════════════════════
# ↓↓↓ 从这里到文件末尾，粘贴，不用看懂 ↓↓↓

def collate_fn(batch):                                        # 把不等长的样本补齐成一批
    m = max(len(x["input_ids"]) for x in batch)                # 这批里最长的那条
    pid = tokenizer.pad_token_id                               # 填充用的编号
    ids, attn, labs = [], [], []                               # 三个容器
    for x in batch:                                            # 逐条补齐
        n = m - len(x["input_ids"])                            # 差多少个补多少个
        ids.append(x["input_ids"] + [pid] * n)                 # 编号补 pad_token_id
        attn.append(x["attention_mask"] + [0] * n)             # 掩码补 0
        labs.append(x["labels"] + [-100] * n)                  # 标签补 -100（补的也不算 loss）
    return {"input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labs, dtype=torch.long)}


# ① 未微调的原始模型
base = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=bnb, device_map="auto")
l1 = eval_loss(base)                                          # 算它的分
print("未微调   loss = %.4f   PPL = %.2f" % (l1, math.exp(l1)))
del base                                                      # 删掉，腾显存
torch.cuda.empty_cache()                                      # 清显存缓存

# ② 微调后 = 原模型 + LoRA 便签
base2 = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=bnb, device_map="auto")
ft = PeftModel.from_pretrained(base2, ADAPTER_DIR)             # ★ 把便签挂回去
l2 = eval_loss(ft)                                            # 算它的分
print("微调后   loss = %.4f   PPL = %.2f" % (l2, math.exp(l2)))
print("loss 变化 = %+.4f" % (l2 - l1))                         # 差多少（负数=变好了）





# ===================================== 问答 =========================================

# ==== 1. 去掉 torch.no_grad() 会怎样？为什么评测不需要它？ ====
# torch.no_grad() 的作用是关闭自动求导。不关的话，PyTorch 会在前向传播时把每一步运算都记进计算图，为反向传播准备梯度。
# 评测时我只要 loss 这个数值，不要梯度，所以这张图建出来是白建 —— 白白占显存，还多花时间。
# 这里有个容易搞错的点：去掉它并不会改变参数。梯度会被算出来、存进每个参数的 .grad 里，但参数更新必须显式调用 optimizer.step() 才会发生，
# 而评测代码里根本没有优化器。所以参数依然是冻结的。总结一句：去不去 no_grad，loss 的结果一样，差别只在显存和速度。

# ==== 2. 为什么 eval 里 labels 直接等于 input_ids，而 d29 的提问部分要标 -100？ ====
# 这两个做的是不同的事，-100 是 label masking，它的作用是决定哪些位置参与 loss 计算 —— HuggingFace 默认忽略标成 -100 的位置。
# 训练时我把提问部分标 -100，是因为推理时提问是用户给的输入，不是模型要生成的内容；让模型去学"怎么复述用户的问题"是浪费，
# 还会干扰它学"给定问题该怎么答"。只让回答部分贡献 loss，模型才专注在回答上，这是训练目标的设计。
# 评测时不标，是因为我要测的是整条序列 —— 提问加回答 —— 整体的语言建模概率，也就是"模型输出这份数据这种风格的概率有多大"。
# 如果屏蔽掉提问，测的就只是"在给定问题条件下的回答部分"，口径变窄了，和我要的"整体像不像"不是一回事。所以差别不在"训练还是评测"，
# 而在"我要让 loss 覆盖哪些位置"：训练是选择性覆盖，评测是全覆盖。
