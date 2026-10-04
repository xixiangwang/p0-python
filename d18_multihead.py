# ========================================= 多头注意力 multihead =========================================

# 多头怎么切、怎么合并（reshape + transpose，你 d16 练过的两个操作在这用）


# ============ 写法 A：循环逐头 ====================
# 好解释，切头用切片、合并用 concatenate
# 知识点：
# ①（np.concatenate(list, axis=-1) 是新东西：把列表里的数组沿最后一维拼起来。axis=-1 你已经在 sum 里用过，含义一样。）
# ② out = out @ Wo中Wo的作用：
#    1.让 8 个头从"并排放着"变成"互相交流"​ ← 最重要
#    2.必须是方阵，保证输出还是 (n, d_model) → 能接残差
#    3.多一层可学习变换，增加表达力

import numpy as np

def softmax(x):
    e = np.exp(x-x.max(axis=-1,keepdims=True))
    return e / e.sum(axis=-1,keepdims=True)

np.random.seed(0)
n,d_model,h = 4,8,2
d_head = d_model // h                      # 4

X = np.random.randn(n,d_model)
Wq = np.random.randn(d_model,d_model) * 0.3
Wk = np.random.randn(d_model,d_model) * 0.3
Wv = np.random.randn(d_model,d_model) * 0.3
Wo = np.random.randn(d_model,d_model) * 0.3

# ① 投影
Q = X @ Wq; K = X @ Wk; V = X @ Wv

# ②③ 循环逐头：切片切头 + 单头注意力
outs = []
for i in range(h):
    lo,hi = i * d_head , (i+1) * d_head
    Qi,Ki,Vi = Q[:,lo:hi],K[:,lo:hi],V[:,lo:hi]      # (n, d_head)
    si = Qi @ Ki.T / np.sqrt(d_head)                 # (n, n)
    wi = softmax(si)                                 # (n, n)
    oi = wi @ Vi                                     # (n, d_head)
    outs.append(oi)

    print("头%d:" % i, Qi.shape, Ki.shape, Vi.shape, "→", si.shape, "→", wi.shape, "→", oi.shape)

# ④ 合并
out = np.concatenate(outs,axis=-1)                  # (n, d_model) 
print("concatenate ->", out.shape)

# ⑤ 输出投影
out = out @ Wo
print("@ Wo        ->", out.shape)



# ================= 写法 B：reshape + transpose —— 工程用的写法 =========
# 知识点：
# ① 新语法：transpose(1, 0, 2)
#   arr.T 是把【所有维度】完全反转；arr.transpose(1,0,2) 是【按你给的顺序重排】。
#   一句话：transpose(参数) 的参数是"新顺序"​ —— (1,0,2) 读作"新的第 0 维取原来的第 1 维，新的第 1 维取原来的第 0 维，第 2 维不动"。
#   为什么需要这一步：reshape(n, h, d_head) 出来的形状是 (n, h, d_head)，"词"在第 0 维；但我们想让**"头"在第 0 维**（这样 @ 才能让每个头独立地做矩阵乘）。​所以要交换前两维。
# ② 多维@
#   @ 在多维时 = 批量矩阵乘。
#   规则：前面的维度当"批次"（相同或广播），最后两维按二维规则算，中间的维度乘完消失。
#   比如(h, n, d_head) @ (h, d_head, n)  ->  (h, n, n)

# 前面一样

# ② 切头（一行代替整个循环）
Qh = Q.reshape(n,h,d_head).transpose(1,0,2)            # (h,n,d_head)
Kh = K.reshape(n,h,d_head).transpose(1,0,2)
Vh = V.reshape(n,h,d_head).transpose(1,0,2)

# ③ 各自算（一次广播，h 个头同时做完）
sc = Qh @ Kh.transpose(0,2,1)                         # (h, n, n)
Wh = softmax(sc / np.sqrt(d_head))                    # (h, n, n)
oh = Wh @ Vh                                          # (h, n, d_head)

# ④ 合并（transpose 一次 + reshape 一次）
outB = oh.transpose(1,0,2).reshape(n,d_model)           # # (n, d_model)，又变化回来

# ⑤ 输出投影
outB = outB @ Wo


print(Qh.shape, sc.shape, Wh.shape, oh.shape, outB.shape)     # (4,4,2) (4,4,4) (4,4,4) (4,4,2) (4,8)
print(np.allclose(out,outB))                    # True