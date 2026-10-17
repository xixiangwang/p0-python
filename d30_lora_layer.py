# -*- coding: utf-8 -*-
"""d30：手写 LoRA 层 —— 把 peft 里那层"便签"自己实现一遍"""

import torch                          # 张量运算 + 自动求导
import torch.nn as nn                 # 神经网络层（Linear / Module）

class LoRALinear(nn.Module):
    """把 nn.Linear 包一层：原权重冻结，旁边挂 A / B 两个小矩阵"""

    def __init__(self,base,r,alpha):           # base=原来的线性层 r=秩 alpha=缩放系数
        super().__init__()

        # ══════════════════ 【抄一遍】★ 接原层 + 冻结 ══════════════════
        # ★ 这是 LoRA 的一半精髓：原权重一点不动
        self.base = base                                    # 存下原来的层（这句同时把它注册成"子模块"）
        for p in self.base.parameters():                    # 遍历原层的每个参数（Linear 有 weight 和 bias 两个）
            p.requires_grad = False                         # ★ 标记"不训练"→ 冻结，训练时永远不更新

        # ══════════════════ 【看懂】取两个维度 ══════════════════
        in_f = base.in_features
        out_f = base.out_features

        # ══════════════════ 【抄一遍】★ 两个小矩阵（核心）══════════════════
        # ★ 这是 LoRA 的另一半精髓：便签本身就这两行
        self.A = nn.Parameter(torch.randn(r,in_f) * 0.01)    # ★ A: (r, in) 用很小的随机数
        self.B = nn.Parameter(torch.zeros(out_f,r))          # ★ B: (out, r) 必须全 0

        # ══════════════════ 【看懂】缩放系数 ══════════════════
        # 你要能讲清：为什么是 alpha/r，不是 alpha
        self.scaling = alpha / r


    # ══════════════════ 【抄一遍】★ 前向传播 ══════════════════
    # ★ 这一段就是"LoRA 是什么"的代码版，面试官会让你写这三行
    def forward(self,x):
        Wx = self.base(x)
        BAx = (x @ self.A.T) @ self.B.T
        return Wx + self.scaling * BAx




# ══════════════════════ 【直接复制】下面全部 ══════════════════════
# ↓↓↓ 从这行开始到文件末尾，全部粘贴，不用看懂 ↓↓↓

torch.manual_seed(0)                            # 固定随机种子，每次结果一样
base = nn.Linear(16, 8)                         # 造一个普通线性层：16 进 8 出
lora = LoRALinear(base, r=4, alpha=8)           # 把它套上 LoRA
x = torch.randn(3, 16)                          # 造 3 条样本，每条 16 维

with torch.no_grad():                           # 关掉梯度记录（这里只检查，不训练）
    print("① 初始态最大误差 =", (lora(x) - base(x)).abs().max().item())
                                                # 比较"套了 LoRA"和"没套"的输出差多少

print("② A 形状 =", tuple(lora.A.shape), " B 形状 =", tuple(lora.B.shape))
                                                # 检查两个小矩阵形状对不对

with torch.no_grad():
    lora.B.copy_(torch.randn(8, 4))              # 把 B 改成非 0，第二条路才有输出
    manual = base(x) + (8 / 4) * ((x @ lora.A.T) @ lora.B.T)
                                                # 按公式手算一遍
    print("③ 公式对照最大误差 =", (lora(x) - manual).abs().max().item())

n_tr  = sum(p.numel() for p in lora.parameters() if p.requires_grad)
                                                # 统计"可训练参数"个数（numel = 元素个数）
n_all = sum(p.numel() for p in lora.parameters())
                                                # 统计"全部参数"个数（含冻结的）
print("④ 可训练 %d / 总共 %d = %.1f%%" % (n_tr, n_all, n_tr / n_all * 100))



