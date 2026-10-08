# ================================= Transformer 块 ================================
import torch
import torch.nn as nn

torch.manual_seed(0)

class Block(nn.Module):
    def __init__(self,d_model,h,d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model,h,bias=False,batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model,d_ff),
            nn.ReLU(),
            nn.Linear(d_ff,d_model),
        )

    def forward(self,x,mask):
        h_in = self.ln1(x)

        # 1.nn.MultiheadAttention的forward(query, key, value, ...)
        # 2.自注意力返回的是 (输出, 权重) 两个东西，我们这里只需要输出
        a,_ = self.attn(h_in,h_in,h_in,attn_mask=mask,need_weights=False)         
                                                        
        x = x + a
        x = x + self.ffn(self.ln2(x))
        return x


# ═══════════════════════════════════════════════════════════
# 【直接复制】跑一次
# ═══════════════════════════════════════════════════════════
d_model, h, d_ff = 64, 4, 256
B, n = 2, 11

block = Block(d_model, h, d_ff)
x = torch.randn(B, n, d_model)                               # 假装是 8.1 出来的输入
mask = torch.triu(torch.ones(n, n), diagonal=1).bool()       # 因果掩码：右上角涂黑

out = block(x, mask)
print("输入 shape =", tuple(x.shape))
print("输出 shape =", tuple(out.shape))
print("形状变了吗？", tuple(x.shape) != tuple(out.shape))
print()
print("这个 block 的参数量 =", sum(p.numel() for p in block.parameters()))





# ========================== 问答 ===========================

#  ===== 1. 为什么 FFN 要先升维再降维？ ======
#   升维是为了给非线性变换更大的空间；降维是为了能接残差。


#  ===== 2. mask 是在 Block 外面算好、传进来的 —— 为什么不在 Block 内部算？（提示：一个模型有 2 个 Block…）===========
#   mask 只跟【序列长度 n】有关，跟模型参数无关 —— 同一个 n 算出来的 mask 永远一样，所以算一次就够。


#  ==== 3. 算block参数 ========
# 层	                                                   参数量
# nn.Linear(in, out)	                                   in × out + out（最后那个 +out 是 bias）
# nn.Embedding(V, D)	                                   V × D
# nn.LayerNorm(D)	                                       2 × D（γ 和 β 各 D 个）
# nn.MultiheadAttention(d, h, bias=False)	               3d × d + d × d
# nn.MultiheadAttention(d, h, bias=True)	               再加 3d + d










