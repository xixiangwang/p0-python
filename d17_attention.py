# ========================================== 注意力直觉 ===============================================

# n        序列长度（几个词）
# d_model  一个词的宽度（原版 512 / Llama-3-8B 4096）
# h        头数（原版 8）
# d_head   每个头内部的宽度 = d_model // h
# d_k      Q/K 的最后一维 = d_head      （单头时 = d_model）
# d_v      V  的最后一维 = d_head      （单头时 = d_model）

# 关系式：  d_model = h × d_head
#           h = 1 时  d_k = d_v = d_model   ← 所以单头看不出区别



# np.round = 逐元素四舍五入。
# np.round(x)        # 取整
# np.round(x, 2)     # 保留 2 位小数

import numpy as np

# 3 个词，每个词 4 维（真实里是 n=512、d=64/512，这里缩小方便看）
np.random.seed(0)
n,d = 3,4

# 所有随机数乘 0.5 —— 让权重分布温和好读（不乘的话会变成 0.999 那种 one-hot）
X = np.random.randn(n,d) * 0.5                 # 3 个词的原始向量
Wq = np.random.randn(d,d) * 0.5
Wk = np.random.randn(d,d) * 0.5
Wv = np.random.randn(d,d) * 0.5

Q = X @ Wq        # (3, 4)
K = X @ Wk        # (3, 4)
V = X @ Wv        # (3, 4)

print("X.shape = ",X.shape,"| Q.shape = ",Q.shape)


# ---------- 第 1 步：相关度 ----------
scores = Q @ K.T

print("scores.shape = ",scores.shape)       # (3, 3)
print(np.round(scores,4))


# ---------- 第 2、3 步：缩放 + softmax ----------
scaled = scores / np.sqrt(d)
W = np.exp(scaled) / np.exp(scaled).sum(axis=-1,keepdims=True)

print("W.shape = ",W.shape)                       # (3, 3)
print(np.round(W,4))
print("每行和 = ",np.round(W.sum(axis=-1),10))    # [1. 1. 1.]


# ---------- 第 4 步：加权求和 ----------
out = W @ V

print("out.shape = ",out.shape)                   # (3, 4)


# ---------- 手算验证：out[0] 应该是 W[0] 对三行 V 的加权和 ----------
# 就是线性代数，但是我忘了
hand = W[0,0] * V[0] + W[0,1] * V[1] + W[0,2] * V[2]

print("手算 = ",np.round(hand,6))
print("代码 = ",np.round(out[0],6))
print("一致 = ",np.allclose(hand,out[0]))         # True
# np.allclose(a, b) = 判断两个数组是否"足够接近"，返回一个 bool
# 为什么需要它 —— 因为浮点数不能直接用 == 比