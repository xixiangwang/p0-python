# ================================= 输入侧 ==========================================

# 字符 → 编号 → 词向量 → 位置向量
# ⭐Embedding分两种：
#   1.是「新建一个对象」，里面装一张 8×4 的表，表里的值是随机生成的，需要自己训练
#   2.就是你以后用 transformers 加载 BERT / Qwen 时拿到的东西 —— 那些词向量表是别人花几万 GPU 小时训出来的，加载进来就有意义，不用从头学。

import torch
import torch.nn as nn

torch.manual_seed(0)
torch.set_printoptions(precision=3,sci_mode=False,linewidth=200)



# ═══════════════════════════════════════════════════════════
# ① 字符 → 编号
# ═══════════════════════════════════════════════════════════
text = "hello world"

chars = sorted(set(text))                   # 去重后排序。1.这里的set知识点补充进了d5.set部分  2.sorted接受任何"可迭代的东西"（set / list / 字符串都行），返回一个排好序的 list
vocab_size = len(chars)                         # 词表大小 = 有几个不同字符
stoi = {ch:i for i,ch in enumerate(chars)}      # 字典，key = 字符，value = 下标序号（编号）
itos = {i:ch for i,ch in enumerate(chars)}      # 字典，key = 下标序号（编号），value = 字符。反查用

ids = [stoi[c] for c in text]                   # 整段文本 → 一串编号

print("字符表 = ",chars)
print("编号 = ",ids)


# ═══════════════════════════════════════════════════════════
# ② 编号 → 整数张量
# ═══════════════════════════════════════════════════════════
idx = torch.tensor(ids,dtype=torch.long)         # 列表 → 张量，元素必须是整数

print("\nidx.shape = ",tuple(idx.shape),"内容 = ",idx)          # tuple(),把任何"可迭代的东西"转成元组。​ 跟 list(...) 是一对



# ═══════════════════════════════════════════════════════════
# ③ 查词向量表：每个编号 → 一个 d_model 维向量
# ═══════════════════════════════════════════════════════════
d_model = 4                                      # 用 4 维方便看（真实模型是 512/768）
tok_emb = nn.Embedding(vocab_size,d_model)       # 建一张 (8, 4) 的表
tok = tok_emb(idx)                               # 拿编号去查表 → (11, 4)

print("\n词向量表 = ") 
print(tok_emb.weight)                            # 把整张表打出来看
                                                 # tok_emb 不是"表"，它是一个"层对象"，表存在它的属性里。所以打印表需要tok_emb.weight
print("\ntok.shape = ",tuple(tok.shape))



# ═══════════════════════════════════════════════════════════
# ④ 查位置向量表，加到词向量上
# ═══════════════════════════════════════════════════════════
n = len(text)
pos_emb = nn.Embedding(n,d_model)          # 注意这里和tok_emb不一样，这里是text的长度
pos = pos_emb(torch.arange(n))
x = tok + pos

print("\npos.shape =", tuple(pos.shape))
print("x = tok + pos 的 shape =", tuple(x.shape))




# =========================== 问答 ======================================

# 1. idx / tok / tok_emb.weight的维度（容易搞混），还是以上面的"hello world"为例
#   idx: 一维，(11,)，11 个整数（还是编号，不是向量）
#   tok: 二维，(11, 4)，11 个向量，每个 4 维
#   tok_emb.weight: 二维，(8, 4)，表本身：8 行（8 个字符）× 4 列

# 2. 为什么最大编号是 len(chars)-1？
#   因为编号是「表的行号」。​ 表有 8 行 → 行号只能是 0~7。

# 3. pos_emb 的表大小为什么设成 n？如果来了一句更长的文本会怎样？
#   位置表是「预先建好、大小固定」的,如果太长就直接截断
