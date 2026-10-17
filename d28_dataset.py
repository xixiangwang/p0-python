# -*- coding: utf-8 -*-
"""d28：构造真实数据集 —— 下载 → 读 → 清洗 → 转 chat → 划分 → 存盘
跑法：cd C:\code\p0-python
      python d28_dataset.py

== 本文件出现的新东西 ==
【要懂】json.load(f) / json.dump(对象, f)  ← 你学过 loads/dumps，load/dump 是"文件版"
【要懂】Alpaca 三字段 · 为什么必须留评测集 · 为什么切之前要先打乱
【直接复制】os.makedirs / os.path.join / urllib.request.urlretrieve
【不用管】print 那几行
[已学过] with open / json.dumps / random.seed / random.shuffle / os.path.exists
        / 列表推导式 / set 去重 / 切片
=========================================================
"""
import json, os, random, urllib.request

DATA_DIR = r"C:\code\p0-python\data"

# ───── ① 下载（33.5 MB · 走 hf-mirror 镜像 · 第二次跑自动跳过）─────
os.makedirs(DATA_DIR, exist_ok=True)                             # 【直接复制】建目录，已存在也不报错
RAW_PATH = os.path.join(DATA_DIR, "alpaca_gpt4_data_zh.json")    # 【直接复制】拼路径
URL = "https://hf-mirror.com/datasets/shibing624/alpaca-zh/resolve/main/alpaca_gpt4_data_zh.json"

if not os.path.exists(RAW_PATH):                                 # ✅ 你学过
    print("下载中（33.5 MB）...")
    urllib.request.urlretrieve(URL, RAW_PATH)                    # 【直接复制】
    print("下载完成")

# ───── ② 读进来 ─────
with open(RAW_PATH, encoding="utf-8") as f:                      # ✅ 你学过
    raw = json.load(f)                                           # ☆ 新：文件 → list

print("总条数 =", len(raw))
print("第一条 =", raw[0])

# ───── ③ 清洗：丢掉空的、去掉重复 ─────
seen, clean = set(), []                                          # ✅ set 你 d13 学过
for ex in raw:
    inst = ex["instruction"].strip()
    out  = ex["output"].strip()
    if not inst or not out:          # 任一为空 → 不要
        continue
    key = (inst, out)                # 完全一样的一对 → 算重复
    if key in seen:
        continue
    seen.add(key)
    clean.append({"instruction": inst,
                  "input": ex.get("input", "").strip(),          # ✅ .get 你 d5 学过
                  "output": out})

print("清洗后 =", len(clean), "（去掉", len(raw) - len(clean), "条）")

# ───── ④ 转成 chat 格式（统一中间表示）─────
def to_chat(ex):
    user = ex["instruction"]
    if ex["input"]:                  # input 不空才拼上去（这个数据集 86% 是空的）
        user = user + "\n" + ex["input"]
    return {"messages": [
        {"role": "user",      "content": user},
        {"role": "assistant", "content": ex["output"]},
    ]}

chat = [to_chat(ex) for ex in clean]                             # ✅ 列表推导式你学过

# ───── ⑤ 打乱 + 切 9:1 ─────
random.seed(42)                      # ✅ 固定种子 → 每次切出来都一样（可复现）
random.shuffle(chat)                 # ✅ 原地打乱

n_eval = int(len(chat) * 0.1)
eval_set  = chat[:n_eval]            # ✅ 切片你学过
train_set = chat[n_eval:]
print("训练集 =", len(train_set), " 评测集 =", len(eval_set))

# ───── ⑥ 存盘（jsonl：一行一条）─────
def save_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:                                            # ✅ 你学过
            f.write(json.dumps(r, ensure_ascii=False) + "\n")     # ✅ dumps 你学过

save_jsonl(os.path.join(DATA_DIR, "train.jsonl"), train_set)
save_jsonl(os.path.join(DATA_DIR, "eval.jsonl"),  eval_set)
print("已存到", DATA_DIR)




# ===================================== 问答 ============================================

# ==== 1. json.load(f) 和 json.loads(text) 差在哪？ ====
#   差一个字母，s 就是 string 的意思。json.loads 收的是字符串，json.load 收的是文件对象。
# 它俩干的是同一件事，json.load(f) 完全等价于 json.loads(f.read())，只是 load 帮你把"打开文件、读出来"这一步省掉了。


# ==== 2. 86% 的 input 是空的 —— 那这个字段存在的意义是什么？ ====
# Alpaca 把一条样本拆成 instruction、input、output 三段。instruction 是任务指令，input 是这条样本对应的具体素材，output 是标准答案。
# 拆开的意义在于把"任务"和"素材"解耦——一个指令模板可以配大量不同素材，既扩充了数据量，也逼模型学任务的通用做法而不是记住某条素材。
# 实际数据里大部分 input 是空的，因为不少任务本身已经完整，不需要额外材料。


# ==== 3. 不打乱、直接取前 90% 当训练集，会出什么问题？ ====
#   先随机打乱再切分，常用 9:1 或 8:1:1。打乱的目的是让训练集和验证集同分布——真实数据往往是有序的，比如按时间爬取、按类别分批生成或从多个来源拼接，
# 直接按顺序切会让验证集的分布和训练集不一致，评测分数就失去意义。另外切分前要固定随机种子，保证划分可复现，不同实验之间才有可比性。
# 这里还有个常见混淆：数据不平衡指单个数据集内部各类比例悬殊，而分布不一致说的是训练集和验证集之间的关系，两者不是一回事。


# ==== 4. random.seed(42) 干什么的？改成 7 会怎样？ ====
#   固定随机种子是为了实验可复现。种子相同，shuffle 的顺序、参数初始化、dropout 的随机掩码都一致；
# 换一个种子，得到的可能是完全不同的数据划分或初始化，所以做对比实验必须在同一种子下进行。