# -*- coding: utf-8 -*-
"""模块 9：ViT（Vision Transformer）—— 把图像也交给 Transformer
跑法：cd C:\code\p0-python && python d24_vit.py

── 本文件出现的新东西 ──────────────────────────────
【要懂】x.permute(0,2,4,1,3,5)  —— transpose 的多维版，一次换好几维
【要懂】nn.Parameter(...)       —— 直接就是一个"可学习的张量"（不是层）
【要懂】x.expand(B, -1, -1)     —— 把长度为 1 的维度放大，不复制内存
✅ 其余（reshape / cat / Linear / LayerNorm / Block / Embedding）都学过
────────────────────────────────────────────────
"""
import torch
import torch.nn as nn

torch.manual_seed(0)


# ═══════════════════════════════════════════════════════════
# ① 图像 → patch 序列
# ═══════════════════════════════════════════════════════════
# 核心目的是：要符合文本(B,n,d_model)的形式，本质是模仿套用
def image_to_patches(img, patch):
    """(B, C, H, W)  →  (B, patch数, 每个patch几个数)"""
    B, C, H, W = img.shape                          # 批次，通道，高，宽
    ph, pw = H // patch, W // patch                 # 横竖各切几块
    x = img.reshape(B, C, ph, patch, pw, patch)     # ① 把 H、W 各拆成 (块数, 块内)
    x = x.permute(0, 2, 4, 1, 3, 5)                 # ② 把"块"的维度挪到前面
    return x.reshape(B, ph * pw, C * patch * patch) # ③ 压平     


# ═══════════════════════════════════════════════════════════
# ② Transformer 块（和模块 8.2 一模一样，直接搬）
# ═══════════════════════════════════════════════════════════
class Block(nn.Module):
    def __init__(self, d_model, h, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, h, bias=False, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))

    def forward(self, x, mask):
        h_in = self.ln1(x)
        a, _ = self.attn(h_in, h_in, h_in, attn_mask=mask, need_weights=False)
        x = x + a
        x = x + self.ffn(self.ln2(x))
        return x


# ═══════════════════════════════════════════════════════════
# ③ ViT：和 mini-GPT 只差「输入怎么来」和「掩码」
# ═══════════════════════════════════════════════════════════
class ViT(nn.Module):
    def __init__(self, img_size=32, patch=8, in_ch=3, d_model=64, h=4, d_ff=256, n_layer=2, n_class=10):
        super().__init__()
        self.patch = patch
        n_patch = (img_size // patch) ** 2                  # 一块能切出几个 patch
        patch_dim = in_ch * patch * patch                   # 每个 patch 拉平后多少个数

        self.proj = nn.Linear(patch_dim, d_model)           # patch → d_model

        # 注意这里是一份，而不是直接B份，因为在__init__里只能出现「跟数据无关的维度」（超参数）,B、n 这类数据维度必须留到 forward 里处理
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model)) # 汇总格（可学习） 
        self.pos_emb = nn.Embedding(n_patch + 1, d_model)   # +1 是给 cls 的

        self.blocks = nn.ModuleList([Block(d_model, h, d_ff) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_class)             # 输出 n_class 个类别分数

    def forward(self, img):
        B = img.shape[0]

        x = image_to_patches(img, self.patch)               # (B, 16, 192)
        x = self.proj(x)                                    # (B, 16, 64)

        # ⭐
        # 在这里才复制B份，-1表示"这一维保持原样"​（不用手算中间那两维是多少）
        # expand这个语法第一次出现(之前学的extend)，只能把长度为 1 的维度放大,但不返回新视图
        # expand 和 repeat 的区别：前者是视图，内存里没真复制。后者是内存里真复制。
        # 所以expand的副作用：因为是视图，不能原地改它（改了会影响所有行）
        cls = self.cls.expand(B, -1, -1)                    # (B, 1, 64)

        # torch.cat([idx, nxt], dim=1)：把几个张量沿某一维"接长"​，返回一个新张量（不改任何输入）
        x = torch.cat([cls, x], dim=1)                      # (B, 17, 64)

        n = x.shape[1]
        pos = self.pos_emb(torch.arange(n, device=img.device))
        x = x + pos                                         # 加位置向量

        mask = torch.zeros(n, n, device=img.device).bool()  # 🔴 全 False：图像不用因果掩码
        for blk in self.blocks:
            x = blk(x, mask)

        x = self.ln_f(x)

        # x[:,0]:    降维，三维直接变二维，(B,64)
        # x[:,0:1]:  保维，(B,1,64) 这里的1是长度
        return self.head(x[:, 0])                           # 只取 cls 那一行 → (B, 10)


# ═══════════════════════════════════════════════════════════
# 跑一次（用随机图片，只为看形状）
# ═══════════════════════════════════════════════════════════
img = torch.randn(2, 3, 32, 32)                             # 2 张 32×32 的彩色图
vit = ViT()
out = vit(img)

print("输入图片 shape =", tuple(img.shape))
print("输出 shape     =", tuple(out.shape), "  ← 10 个类别的分数")
print("参数量         =", sum(p.numel() for p in vit.parameters()))





# ====================== 问答 ===========================

# ==== 1.为什么必须有 permute 这一步？ 如果只有两个 reshape、去掉 permute，会怎样？ ===
# reshape 只能合并「内存里连续」的维度 —— 它是按存储顺序从头切的。



# ==== ⭐2. ViT 完全没用卷积 —— 那它靠什么"看懂"图像的局部结构？（提示：想想 attention 在什么范围内做加权） ====
# ViT 靠两件事"看懂"图像：

# ① patch 内部（局部信息）
# patch 拉平时，像素的排列顺序保留了（左上、右上、左下、右下）。
# 后面的 nn.Linear 能学到"这些位置的哪种组合有意义" —— 相当于一个很小的局部模式识别器。

# ② patch 之间（全局信息）
# attention 让每个 patch 都能和所有其他 patch 交流 → 能学到"左上角的块和右下角的块有什么关系"。
# 这比卷积的局部感受野更全局 —— 卷积要堆很多层才能"看到"远处，attention 一步就能。

# 代价是：patch 数 n 决定计算量（O(n²)​）。所以 patch 切得越小、细节越多，但越慢。



# ==== 3.把 patch 从 8 改成 16 —— patch 数变几个？每个 patch 拉平后多少个数？序列长度变多长？ ====
#这里需要注意的是序列长度，还需要加上一个cls




