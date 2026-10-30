import numpy as np

def softmax(x):
    e = np.exp(x-x.max(axis=-1,keepdims=True))
    return e / e.sum(axis=-1,keepdims=True)

def mha_written(x, Wq, Wk, Wv, Wo, h):
    B,n,d_model = x.shape
    d_head = d_model // h

    Q = x @ Wq         # (B,n,d_model)
    K = x @ Wk
    V = x @ Wv

    Qh = Q.reshape(B,n,h,d_head).transpose(0,2,1,3)
    Kh = K.reshape(B,n,h,d_head).transpose(0,2,1,3)
    Vh = V.reshape(B,n,h,d_head).transpose(0,2,1,3)

    scores = Qh @ Kh.transpose(0,1,3,2) / np.sqrt(d_head)
    A = softmax(scores)
    out = (A @ Vh).transpose(0,2,1,3).reshape(B,n,d_model)

    return out @ Wo, A 

# 测试
if __name__ == "__&#8203;main__":
    d_model, h = 8, 2
    rng = np.random.default_rng(0)
    Wq = rng.normal(0, 0.1, (d_model, d_model))
    Wk = rng.normal(0, 0.1, (d_model, d_model))
    Wv = rng.normal(0, 0.1, (d_model, d_model))
    Wo = rng.normal(0, 0.1, (d_model, d_model))
    x = np.random.default_rng(1).normal(0, 1, (2, 4, d_model))

    out, A = multi_head_attention(x, Wq, Wk, Wv, Wo, h)
    print("out.shape   =", out.shape)
    print("out[0,0,:4] =", np.round(out[0, 0, :4], 6))
    print("A[0,0,0,:]  =", np.round(A[0, 0, 0, :], 6))
    print("A 每行和     =", np.round(A.sum(-1).ravel()[:4], 6))
