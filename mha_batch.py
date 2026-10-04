# ============ 先填这张表（填完再写代码）============
# B = batch size（一次喂几条样本）
#
#   步骤        变量                        形状
#   输入        X                        (B, n, d_model)
#   ①           Q / K / V                (B, n, d_model)
#   ②           reshape 后                (B, n, h, d_head)
#   ②           Qh / Kh / Vh（转置后）      (B, h, n, d_head)
#   ③           sc                       (B, h, n, n)
#   ③           Wh                       (B, h, n, n)
#   ③           oh                       (B, h, n, d_head)
#   ④           合并后 out                (B, n, d_model)
#   ⑤           out @ Wo                 (B, n, d_model)
#
# 对比：非 batch 版是 (n, d_model) → (h, n, d_head) → (h, n, n) → (n, d_model)
#       batch 版就是在【最前面多一维 B】，那一维自始至终不动


import numpy as np

def softmax(x):
    e = np.exp(x - x.max(axis=-1,keepdims=True))
    return e / e.sum(axis=-1,keepdims=True)

def mha_batch(X,Wq,Wk,Wv,Wo,h):
    B,n,d_model = X.shape
    assert d_model % h == 0,"d_model必须被h整除,实际d_model=%d,h=%d" % (d_model,h)
    d_head = d_model // h

    # 投影
    Q = X @ Wq; K = X @ Wk; V = X @ Wv
    assert Q.shape == (B, n, d_model), Q.shape

    # 切头
    Qh = Q.reshape(B,n,h,d_head).transpose(0,2,1,3)              # (B,h,n,d_head)
    Kh = K.reshape(B,n,h,d_head).transpose(0,2,1,3)
    Vh = V.reshape(B,n,h,d_head).transpose(0,2,1,3)

    assert Qh.shape == (B, h, n, d_head), Qh.shape

    # 缩放点积
    sc = Qh @ Kh.transpose(0,1,3,2)                              # (B,h,n,n)
    Wh = softmax(sc / np.sqrt(d_head))
    oh = Wh @ Vh                                                 # (B,h,n,d_head)


    assert sc.shape == (B, h, n, n), sc.shape
    assert Wh.shape == (B, h, n, n), Wh.shape
    assert oh.shape == (B, h, n, d_head), oh.shape

    # 合并
    out = oh.transpose(0,2,1,3).reshape(B,n,d_model)

    assert out.shape == (B, n, d_model), out.shape

    # 输出投影
    out = out @ Wo
    assert out.shape == (B, n, d_model), out.shape


    return out





