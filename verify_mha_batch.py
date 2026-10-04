# -*- coding: utf-8 -*-
"""零修改验收脚本：verify_mha_batch.py
跑法：cd C:\\code\\p0-python && python verify_mha_batch.py

前提：mha_batch.py 里存在 def mha_batch(X, Wq, Wk, Wv, Wo, h)
"""
import numpy as np
from mha_batch import mha_batch


# ---------------- 参照实现（非 batch 版，自包含）----------------
def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def mha_single(X, Wq, Wk, Wv, Wo, h):
    n, d_model = X.shape
    d_head = d_model // h
    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv
    Qh = Q.reshape(n, h, d_head).transpose(1, 0, 2)
    Kh = K.reshape(n, h, d_head).transpose(1, 0, 2)
    Vh = V.reshape(n, h, d_head).transpose(1, 0, 2)
    Wh = softmax((Qh @ Kh.transpose(0, 2, 1)) / np.sqrt(d_head))
    oh = Wh @ Vh
    return oh.transpose(1, 0, 2).reshape(n, d_model) @ Wo


# ---------------- 验收 ----------------
CASES = [(1, 4, 8, 2), (3, 4, 8, 4), (2, 6, 12, 1),
         (2, 6, 12, 3), (4, 5, 16, 4), (1, 3, 8, 8), (2, 7, 16, 2)]
#        (B, n, d_model, h)

np.random.seed(0)
fails = 0

for trial, (B, n, d_model, h) in enumerate(CASES):
    X  = np.random.randn(B, n, d_model)
    Wq = np.random.randn(d_model, d_model) * 0.3
    Wk = np.random.randn(d_model, d_model) * 0.3
    Wv = np.random.randn(d_model, d_model) * 0.3
    Wo = np.random.randn(d_model, d_model) * 0.3

    out = mha_batch(X, Wq, Wk, Wv, Wo, h)
    ref = np.stack([mha_single(X[b], Wq, Wk, Wv, Wo, h) for b in range(B)])

    shape_ok = (out.shape == (B, n, d_model))
    close_ok = np.allclose(out, ref)
    nan_ok   = not np.isnan(out).any()
    if not (shape_ok and close_ok and nan_ok):
        fails += 1

    print("第%d组: B=%d n=%d d_model=%d h=%d -> out%s  形状=%s 一致=%s 无nan=%s"
          % (trial, B, n, d_model, h, out.shape, shape_ok, close_ok, nan_ok))

print()
print("错 %d 组" % fails)
