# ====================================== 位置编码 =============================================

# 知识点：
#   1.a[起 : 止 : 步],a[0::1]就是从0开始，步长为2，所以就是取下标为偶数的
#   2.np.arange(n) 就是 np.array(list(range(n))) 的简写
#   3.a.[:,None] 意思是把以为变成二维，为了广播规则
# 


import numpy as np

np.set_printoptions(precision=4, suppress=True, linewidth=200)
np.random.seed(0)


def pos_enc(n,d_model):
    pos = np.arange(n)[:,None]               # (n, 1)      位置序号 0..n-1
    i = np.arange(0,d_model,2)[None,:]       # (1, d/2)    0, 2, 4, ... 偶数列下标
    denom = 10000 ** (i / d_model)           # (1, d/2)    每对维度的分母
    angles = pos / denom                     # (n, d/2)    广播：(n,1)/(1,d/2)
    pe = np.zeros((n,d_model))
    pe[:,0::2] = np.sin(angles)              # 偶数列 = sin
    pe[:,1::2] = np.cos(angles)              # 奇数列 = cos
    return pe




# =========================== ① 基本形状与数字 ===========================下面代码先不用看，最后的问题需要看
print("=== ① 基本形状与数字 ===")
d_model, n = 8, 5
PE = pos_enc(n, d_model)
print("PE.shape =", PE.shape, " ← (n, d_model)")
print()
print("各对维度的分母 10000^(2i/d_model)：")
for i in range(0, d_model, 2):
    print("  第 %d 对(i=%d): 分母 = %10.4f" % (i // 2, i // 2, 10000 ** (i / d_model)))
print()
print("PE =")
print(PE)
print()
print("位置 0 的编码 =", np.round(PE[0], 4).tolist(), " ← 恒为 [0,1,0,1,...]")

# =========================== ② 周期的直观 ===========================
print()
print("=== ② 各列'变化快慢'不一样（这是设计的核心）===")
print("  列 0（分母 1    ，周期 6.28）  :", np.round(PE[:, 0], 4).tolist())
print("  列 6（分母 1000 ，周期 6283）  :", np.round(PE[:, 6], 4).tolist())
print("  → 列 0 几乎每个位置都变；列 6 五个位置几乎一样（像'万位数字'）")

# =========================== ③ 关键性质：相对位置可线性表达 ===========================
print()
print("=== ③ 关键性质：PE(pos+k) 能由 PE(pos) 乘一个固定矩阵得到 ===")
PE_big = pos_enc(20, d_model)
for k in (1, 2, 3):
    A = np.vstack([PE_big[p] for p in range(5)])           # PE(pos)   pos=0..4
    B = np.vstack([PE_big[p + k] for p in range(5)])       # PE(pos+k)
    M, *_ = np.linalg.lstsq(A, B, rcond=None)              # 解出 M
    print("  k=%d: 用同一个 M 预测所有 pos 的最大误差 = %.2e" % (k, np.abs(A @ M - B).max()))
print("  → 误差 1e-15 = 机器精度，等于精确成立。所以 attention 能学到'往前看 k 个'")

# =========================== ④ 没有位置编码会怎样 ===========================
print()
print("=== ④ 反证：周期序列里，没有 PE 就分不清'第几个 A' ===")
A = np.random.randn(d_model)
B = np.random.randn(d_model)
X = np.stack([A, B, A, B, A, B])                           # 同一个词 A 在位置 0/2/4
Xp = X + pos_enc(6, d_model)                               # 加上 PE

print("  X 里位置 0 / 2 / 4 是同一个向量 A：", np.allclose(X[0], X[2]), np.allclose(X[0], X[4]))
print("  加上 PE 后位置 0 / 2 / 4 还一样吗？  ",
      np.allclose(Xp[0], Xp[2]), np.allclose(Xp[0], Xp[4]), " ← False，位置信息被注入")
print("  PE[0] =", np.round(PE[0], 4).tolist())
print("  PE[2] =", np.round(pos_enc(6, d_model)[2], 4).tolist())
print("  → 同一个词 + 不同位置编码 = 不同的输入，模型才分得清")





# ======================================= 两问 ===================================

# 1.位置编码为什么必须用「多档不同频率」？

# sin/cos 是会【转圈】的。一根针转满一圈（360°）就回到起点，
# 这时它区分不出「转了一圈」和「压根没转」——位置信息丢了。
#
# 【具体数字】以「分母 1」那一档为例（位置每 +1 转 1 弧度，周期 2π≈6.28）：
#   位置 0 与位置 6 的编码距离 = 0.28
#   位置 0 与位置 1 的编码距离 = 0.96
#   → 隔 6 个位置反而比隔 1 个位置更"近"！这就是撞车。
#
# 【所以要多档】转得慢的针还没转满，能接棒继续区分：
#   位置 6 时：第 1 档已经转了 95%（快要回到原点，看不出差别）
#             第 2 档只转了 9.5%（差得很明显）
#   两档合起来 → 位置 0 与位置 6 的距离变成 0.66，拉开了。


# 2.RoPE（旋转位置编码）是什么？和 sin/cos 版差在哪？

# 原版（绝对位置编码）是把位置编码【加】到输入上：X = 词向量 + PE(pos)
# RoPE 是【旋转】Q 和 K：按位置把 Q、K 的每个二维分量对转一个角度。