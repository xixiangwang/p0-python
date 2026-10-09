# 就是自己单独写一遍的d18
# 知识点：
# ① assert语法：
#   assert 条件                 不满足就报错
#   assert 条件, "错误消息"      不满足就报错，并带上消息

# ② ⭐Wo的作用
#   Wo 是「让多个头之间交流」的那一步


import numpy as np

def softmax(x):
    e = np.exp(x - x.max(axis=-1,keepdims=True))
    return e / e.sum(axis=-1,keepdims=True)

def multi_head_attention(X,Wq,Wk,Wv,Wo,h):
    n,d_model = X.shape
    assert d_model % h == 0, 'd_model 必须能被 h 整除，实际 d_model=%d, h=%d' % (d_model, h)
    d_head = d_model // h

    # 投影
    Q = X @ Wq; K = X @ Wk; V = X @ Wv
    assert Q.shape == (n,d_model),Q.shape


    # 切头
    Qh = Q.reshape(n,h,d_head).transpose(1,0,2)          # (h,n,d_head)
    Kh = K.reshape(n,h,d_head).transpose(1,0,2)
    Vh = V.reshape(n,h,d_head).transpose(1,0,2)

    assert Qh.shape == (h,n,d_head),Qh.shape

    # 缩放点积注意力
    sc = Qh @ Kh.transpose(0,2,1)                        # (h,n,n)
    Wh = softmax(sc / np.sqrt(d_head))
    oh = Wh @ Vh                                         # (h,n,d_head)

    assert sc.shape == (h,n,n),sc.shape
    assert oh.shape == (h,n,d_head),oh.shape

    # 合并
    out = oh.transpose(1,0,2).reshape(n,d_model)         # (n,d_model)

    assert out.shape == (n,d_model),out.shape

    # 输出投影
    out = out @ Wo
    assert out.shape == (n,d_model),out.shape

    return out




# ==================== 验收 ====================
# 直接复制的

def mha_loop(X, Wq, Wk, Wv, Wo, h):
    """参照实现：逐头循环（等价于 d18 写法 A）"""
    n, d_model = X.shape
    d_head = d_model // h
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    outs = []
    for i in range(h):
        lo, hi = i * d_head, (i + 1) * d_head
        Qi, Ki, Vi = Q[:, lo:hi], K[:, lo:hi], V[:, lo:hi]
        outs.append(softmax(Qi @ Ki.T / np.sqrt(d_head)) @ Vi)
    return np.concatenate(outs, axis=-1) @ Wo


np.random.seed(0)
fails = 0
for trial in range(5):
    n      = int(np.random.randint(2, 9))
    d_model = int(np.random.choice([8, 12, 16]))
    h      = int(np.random.choice([x for x in [1, 2, 3, 4, 8] if d_model % x == 0]))

    X  = np.random.randn(n, d_model)
    Wq = np.random.randn(d_model, d_model) * 0.3
    Wk = np.random.randn(d_model, d_model) * 0.3
    Wv = np.random.randn(d_model, d_model) * 0.3
    Wo = np.random.randn(d_model, d_model) * 0.3

    a = multi_head_attention(X, Wq, Wk, Wv, Wo, h)
    b = mha_loop(X, Wq, Wk, Wv, Wo, h)
    ok = (a.shape == b.shape) and np.allclose(a, b) and not np.isnan(a).any()
    if not ok:
        fails += 1
    print("第%d组: n=%-2d d_model=%-3d h=%-2d  一致=%s" % (trial, n, d_model, h, ok))

print("错 %d 组" % fails)
