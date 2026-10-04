# -*- coding: utf-8 -*-
"""模块 5：把 numpy 手写 MHA 与 torch 的 nn.MultiheadAttention 数值对齐
跑法：cd C:\\code\\p0-python && python torch_compare.py
"""
import numpy as np
import torch
import torch.nn as nn

from mha_batch import mha_batch


torch.manual_seed(0)
np.random.seed(0)

d_model, h = 8, 2
B, n = 3, 5

# ---------- ① 建 torch 的 MHA ----------
#   bias=False       → torch 默认带 bias，我们的 numpy 版没有，所以关掉
#   batch_first=True → 输入变成 (B, n, d_model)，和我们的布局一致（省掉转置）
mha = nn.MultiheadAttention(d_model, h, bias=False, batch_first=True)
mha.eval()

print("=== torch 内部的权重形状 ===")
print("  in_proj_weight  :", tuple(mha.in_proj_weight.shape),
      " ← (3*d_model, d_model)，Q/K/V 竖着拼成一块")
print("  out_proj.weight :", tuple(mha.out_proj.weight.shape))

# ---------- ② 从 torch 读出权重，转成 numpy 侧约定 ----------
W_ip = mha.in_proj_weight.detach().numpy()      # (3*d_model, d_model)
Wo_t = mha.out_proj.weight.detach().numpy()     # (d_model, d_model)

# torch 算的是 X @ W.T；我们算的是 X @ W  →  所以取 .T
Wq = W_ip[0:d_model].T
Wk = W_ip[d_model:2*d_model].T
Wv = W_ip[2*d_model:3*d_model].T
Wo = Wo_t.T

print()
print("=== 搬过来的 numpy 权重 ===")
print("  Wq.shape =", Wq.shape, " Wk.shape =", Wk.shape,
      " Wv.shape =", Wv.shape, " Wo.shape =", Wo.shape)

# ---------- ③ numpy 侧算 ----------
X = np.random.randn(B, n, d_model)
out_np = mha_batch(X, Wq, Wk, Wv, Wo, h)

# ---------- ④ torch 侧算同一个输入 ----------
with torch.no_grad():
    Xt = torch.from_numpy(X).float()
    out_t, attn_t = mha(Xt, Xt, Xt, need_weights=True, average_attn_weights=False)
out_t = out_t.detach().numpy()
attn_t = attn_t.detach().numpy()

# ---------- ⑤ 对拍 ----------
print()
print("=== 对拍 ===")
print("  numpy out.shape =", out_np.shape)
print("  torch out.shape =", out_t.shape)
print()
print("  numpy 第 0 个样本第 0 行 =", np.round(out_np[0, 0], 6).tolist())
print("  torch 第 0 个样本第 0 行 =", np.round(out_t[0, 0], 6).tolist())
print()
print("  🔴 最大绝对误差 = %.3e" % np.abs(out_np - out_t).max())
print("  🔴 一致 (atol=1e-5) =", np.allclose(out_np, out_t, atol=1e-5))
print()
print("=== 顺便：torch 的注意力权重 ===")
print("  attn_weights.shape =", attn_t.shape, " ← (B, h, n, n)，和你算的 Wh 同形")
