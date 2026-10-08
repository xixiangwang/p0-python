# ==================================== 残差连接resnet 和 层归一化 layernorm ====================================

# LayerNorm 归一化的是「最后一个维度 d_model」，不是 batch 维、也不是序列维。

import numpy as np

np.set_printoptions(precision=4, suppress=True, linewidth=200)   # 【不用管】显示设置
np.random.seed(0)

def layernorm(x,gamma,beta,eps=1e-5):
    mu = x.mean(axis=-1,keepdims=True)         # ① 每个 token 自己的均值
    var = x.var(axis=-1,keepdims=True)         # ② 每个 token 自己的方差
    y = (x-mu) / np.sqrt(var + eps)            # ③ 归一化（+eps 防除零）
    return gamma * y + beta                    # ④ 可学习的缩放 + 平移



# ══════════════════════════════════════════════════════════════
# 实验 1：归一化后 均值≈0、方差≈1
# ══════════════════════════════════════════════════════════════
print("=== 实验 1 ===")
B, n, d_model = 2, 3, 4
x = np.random.randn(B, n, d_model) * 5 + 3       # 故意搞乱：均值3、标准差5
gamma = np.ones(d_model)
beta  = np.zeros(d_model)

out = layernorm(x, gamma, beta)
print("  前 x[0,0] =", np.round(x[0,0],4).tolist(),
      " 均值 %.4f 方差 %.4f" % (x[0,0].mean(), x[0,0].var()))
print("  后 y[0,0] =", np.round(out[0,0],4).tolist(),
      " 均值 %.4f 方差 %.4f" % (out[0,0].mean(), out[0,0].var()))
print("  → 每个 token 都被拉成 均值0 / 方差1")

# ══════════════════════════════════════════════════════════════
# 实验 2：手算 vs 机器算
# ══════════════════════════════════════════════════════════════
print()
print("=== 实验 2 ===")
v = x[0, 0]
mu = v.mean()
var = ((v - mu) ** 2).mean()
hand = (v - mu) / np.sqrt(var + 1e-5)
print("  均值 = %.6f   方差 = %.6f" % (mu, var))
print("  手算 =", np.round(hand,4).tolist())
print("  机器 =", np.round(out[0,0],4).tolist())
print("  一致 =", bool(np.allclose(hand, out[0,0])))

# ══════════════════════════════════════════════════════════════
# 实验 3 ★★★ 证明「归一化的是最后一维」：换 batch / 换序列长度，结果不变
# ══════════════════════════════════════════════════════════════
print()
print("=== 实验 3 ★ 关键：跟 batch、跟序列长度都无关 ===")
x_big = np.random.randn(7, 11, d_model)   # 换成 B=7, n=11
x_big[0, 0] = x[0, 0]                     # 只把第一个 token 设成一样的
out_big = layernorm(x_big, gamma, beta)
print("  B=2, n=3   →", np.round(out[0,0],4).tolist())
print("  B=7, n=11  →", np.round(out_big[0,0],4).tolist())
print("  一致 =", bool(np.allclose(out[0,0], out_big[0,0])))
print("  → 同一个 token 的结果完全不受 batch 大小、序列长度影响")
print("  → 因为归一化只在【自己那 d_model 个数】之内做")

# ══════════════════════════════════════════════════════════════
# 实验 4：eps 是干什么的
# ══════════════════════════════════════════════════════════════
print()
print("=== 实验 4：eps ===")
flat = np.array([[2.0, 2.0, 2.0, 2.0]])   # 四个数全一样 → 方差 0
with np.errstate(invalid='ignore', divide='ignore'):
    print("  不加 eps (x-mu)/sqrt(var)      =", (flat-flat.mean())/np.sqrt(flat.var()))
print("  加了 eps (x-mu)/sqrt(var+1e-5)  =", layernorm(flat, np.ones(4), np.zeros(4)))
print("  → 不加 eps 会 0/0 = nan，加了得 0，安全")

# ══════════════════════════════════════════════════════════════
# 实验 5：gamma / beta 的作用
# ══════════════════════════════════════════════════════════════
print()
print("=== 实验 5：gamma / beta 让模型能「反悔」 ===")
y1 = layernorm(x[0,0].reshape(1,-1), np.ones(4),  np.zeros(4))[0]
print("  gamma=1, beta=0（初始化）→", np.round(y1,4).tolist(),
      " 均值 %.4f 方差 %.4f" % (y1.mean(), y1.var()))
y2 = layernorm(x[0,0].reshape(1,-1), np.full(4,2.0), np.full(4,5.0))[0]
print("  gamma=2, beta=5（训练后）→", np.round(y2,4).tolist(),
      " 均值 %.4f 方差 %.4f" % (y2.mean(), y2.var()))
print("  → 归一化只是「起点」，gamma/beta 让模型自己决定要什么分布")

# ══════════════════════════════════════════════════════════════
# 实验 6：残差为什么能治梯度消失
# ══════════════════════════════════════════════════════════════
print()
print("=== 实验 6：残差 ===")
print("  一层 y=x+f(x) → dy/dx = 【1 + f'(x)】展开成 2 项")
print("  两层 → 展开成 4 项，其中 1 项是 1×1（完全不经过任何 f）")
print("  L 层 → 展开成 2^L 项，【恒等路径】那一项系数恒为 1")
print("  → 反向传播时梯度总有一条「不衰减的直通路径」，所以不会归零")
print("  对比：没残差时梯度 = f'(x_L)·…·f'(x_1)，全是乘法，任一层小就整体变小")

# ══════════════════════════════════════════════════════════════
# 实验 7：axis 写错会怎样（一眼看出 LayerNorm 和 BatchNorm 的区别）
# ══════════════════════════════════════════════════════════════
print()
print("=== 实验 7：axis=-1 和 axis=0 的区别 ===")
print("  axis=-1 的均值 shape :", x.mean(axis=-1, keepdims=True).shape,
      " <- 每个 token 一个均值  = LayerNorm")
print("  axis=0  的均值 shape :", x.mean(axis=0, keepdims=True).shape,
      " <- 每个维度一个均值     = BatchNorm")




# 

# 1.残差为什么能治梯度消失？
#   残差让梯度展开后存在一条系数恒为 1 的恒等路径，所以无论多深，梯度都不会被连乘衰减到 0。