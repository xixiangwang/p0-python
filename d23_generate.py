# -*- coding: utf-8 -*-
"""模块 8.3b：生成（自回归采样）—— 用假模型，不用训练
跑法：cd C:\code\p0-python && python d23_generate.py

── 本文件出现的新东西 ──────────────────────────────
【要懂】torch.multinomial(probs, 1)   —— 按概率抽样（生成的核心）
【要懂】torch.cat([a, b], dim=1)      —— 沿第 1 维拼接
【要懂】model.eval() / model.train()  —— 切推理/训练模式
【要懂】torch.no_grad()               —— 不建计算图，省内存
✅ 其余（zeros / for / 下标 / % / softmax）都能看懂，没有新的
────────────────────────────────────────────────
"""
import torch
import torch.nn.functional as F

torch.manual_seed(0)


# ═══════════════════════════════════════════════════════════
# 假模型：下一个字符 = 当前字符 + 1（到 10 归零）
#   真的 MiniGPT 的分数是"训出来的"，这里用写死的规则代替
#   好处：不用训练、几秒跑完，专注看【采样循环】
# ═══════════════════════════════════════════════════════════
def fake_model(idx):
    B, n = idx.shape                    # idx 形状 (1, n)
    logits = torch.zeros(B, n, 10)      # 先建一个全 0 的分数表
    for i in range(n):                  # 逐个位置处理
        cur = int(idx[0, i])            #   这个位置上是哪个字符
        nxt = (cur + 1) % 10            #   它的"下一个"应该是谁
        logits[0, i, nxt] = 10.0        #   给"下一个"打高分
    return logits                       # 返回 (1, n, 10)


# ═══════════════════════════════════════════════════════════
# 自回归采样：一次生成一个字符，接到末尾，再生成下一个
# ═══════════════════════════════════════════════════════════
idx = torch.tensor([[0]])                        # 起点：只有 1 个字符
print("起点 idx =", idx.tolist(), " shape =", tuple(idx.shape))

with torch.no_grad():                            # 推理不用算梯度
    for step in range(10):
        logits_all = fake_model(idx)             # (1, n, 10)
        logits_last = logits_all[:, -1, :]       # (1, 10) 只要【最后一个位置】
        probs = F.softmax(logits_last, dim=-1)   # 分数 → 概率
        nxt = torch.multinomial(probs, 1)        # 按概率抽一个
        idx = torch.cat([idx, nxt], dim=1)       # 接到末尾
        print("  step%d  idx %-26s 抽到 %s" % (step, tuple(idx.shape), nxt.tolist()))

print("\n最终 =", idx.tolist())

