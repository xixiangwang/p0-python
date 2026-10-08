import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# ═══════════════════════════════════════════════════════════════════
# 【直接复制】1. 数据（字符级 tokenizer）
# ═══════════════════════════════════════════════════════════════════
TEXT = """Deep learning changed how we build software. A model is trained on many examples, and it learns patterns from them. Attention lets a model look at all the words at once and decide which ones matter. A transformer stacks many such blocks. Each block has two parts: one that mixes information across positions, and one that processes each position alone. Residual connections keep the gradient alive during training. Layer normalisation keeps the numbers calm. Together they make a deep network trainable.
""" * 25

chars = sorted(set(TEXT))                        # 所有出现过的字符，去重排序
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}     # 字符 → 编号
itos = {i: ch for i, ch in enumerate(chars)}     # 编号 → 字符
data = torch.tensor([stoi[c] for c in TEXT], dtype=torch.long)   # 整篇 → 一串编号


def get_batch(batch_size, block_size):
    """随机抽 batch_size 段长度 block_size 的片段"""
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])   # 目标 = 输入后移一位
    return x, y


# ═══════════════════════════════════════════════════════════════════
# 【要你写】2. Transformer 块
# ═══════════════════════════════════════════════════════════════════
class Block(nn.Module):
    def __init__(self,d_model,h,d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model,h,bias=False,batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model,d_ff),
            nn.ReLU(),
            nn.Linear(d_ff,d_model)
            )

    def forward(self,x,mask):
        h_in = self.ln1(x)
        a,_ = self.attn(h_in,h_in,h_in,attn_mask=mask,need_weights=False)
        x = x + a

        x = x + self.ffn(self.ln2(x))              # pre-LN(现阶段都是)

        return x


# ═══════════════════════════════════════════════════════════════════
# 【要你写】3. 整个模型
# ═══════════════════════════════════════════════════════════════════
class MiniGPT(nn.Module):
    def __init__(self,vocab_size,d_model,h,d_ff,n_layer,max_len):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size,d_model)
        self.pos_emb = nn.Embedding(max_len,d_model)
        self.blocks = nn.ModuleList([Block(d_model,h,d_ff) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model,vocab_size)

    def forward(self,idx):
        B,n = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(n,device=idx.device))
        x = tok + pos


        mask = torch.triu(torch.ones(n, n, device=idx.device), diagonal=1).bool()
        for blk in self.blocks:
            x = blk(x, mask)
        x = self.ln_f(x)
        return self.head(x)      # (B, n, vocab_size)



# ═══════════════════════════════════════════════════════════════════
# 【直接复制】4. 训练（PyTorch 五步）
# ═══════════════════════════════════════════════════════════════════
d_model, h, d_ff = 64, 4, 256
n_layer, max_len, batch_size = 2, 64, 32

model = MiniGPT(vocab_size, d_model, h, d_ff, n_layer, max_len)
print("词表大小 %d · 参数量 %d" % (vocab_size, sum(p.numel() for p in model.parameters())))

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

for step in range(2001):
    x, y = get_batch(batch_size, max_len)

    logits = model(x)                                                     # ① 前向
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1)) # ② 损失
    optimizer.zero_grad()                                                 # ③ 清梯度
    loss.backward()                                                       # ④ 反向
    optimizer.step()                                                      # ⑤ 更新

    if step % 200 == 0:
        print("  step %4d   loss = %.4f" % (step, loss.item()))


# ═══════════════════════════════════════════════════════════════════
# 【直接复制】5. 生成（自回归采样）
# ═══════════════════════════════════════════════════════════════════
def generate(model, n_new, max_len):
    model.eval()                                   # 推理模式
    idx = torch.zeros((1, 1), dtype=torch.long)    # 从 0 号字符起步
    with torch.no_grad():                          # 推理不用算梯度
        for _ in range(n_new):
            logits = model(idx[:, -max_len:])[:, -1, :]      # 只要最后一个位置的输出
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)   # 按概率抽一个
            idx = torch.cat([idx, nxt], dim=1)
    model.train()                                  # 切回训练模式
    return idx

print("\n=== 生成 300 个字符 ===")
out = generate(model, 300, max_len)
print("".join(itos[i] for i in out[0].tolist()))