# p0-python · 大模型应用算法求职准备（学习 + 实验代码）

从 Python 基础一路写到 **QLoRA 微调与四层评测** 的个人实验仓库。
**核心产出是 A2 阶段的微调 / 评测项目，完整技术报告见 [`REPORT_A2_v1.md`](REPORT_A2_v1.md)。**

> 全部实验在 **单卡 4GB 显存**（RTX 3050 Ti Laptop）上完成，基座模型 `Qwen2.5-0.5B-Instruct`（494M 参数）。

---

## 三个阶段的代码

| 阶段 | 内容 | 代表脚本 |
|---|---|---|
| **P0** Python 迁移 | C → Python 语法迁移、数据结构手写、LeetCode 入门 | `d1_*`–`d13_*` · `ds_*.py` · `lc_*.py` |
| **A1** Transformer 从零实现 | numpy 手写注意力 / 多头 / 位置编码 / Block / ViT / mini-GPT | `mha_numpy.py` · `d17`–`d24` · `mini_gpt.py` |
| **A2** 微调与评测 | 数据构造 → QLoRA → 多口径评测 → 误差归因 → 数据迭代 → 消融 | `d26`–`d39` · `REPORT_A2_v1.md` |

**A1 的关键验证**：numpy 手写的多头注意力与 `torch.nn.MultiheadAttention` 逐元素对齐，**最大误差 8.9e-08**（单样本与 batch 两种输入都验证过）。

---

## A2 核心结果（详细分析见报告）

| 指标 | 结果 |
|---|---|
| 留出集困惑度 | PPL **19.63 → 15.97**（loss 降 0.2064） |
| 人工 rubric（25 题盲评） | **34 : 33** —— PPL 下降但人工评分持平 |
| 领域结构化抽取（同分布） | 全字段正确率 **0% → 100%** |
| 分布外 OOD-A（110 条） | **90.9%**；误差归因 + 改数据后复测 **→ 100%** |
| 独立终审集（100 条，全新语义） | **72.0%**（规范化口径）；补数据后 **94.0%** |
| 能力边界 | "薪资面议→null" 等格式类 **100%**；"4万→40千" 单位换算 **0%** |

**两个消融**（`d32` / `d39`）：

- **LoRA 秩** `r` = 4 / 8 / 16 → 评测 loss **2.7555 / 2.7705 / 2.7767** 单调上升。逐条配对检验 **t = 13.18（175/200 条同向）**，但换算到 PPL 仅差 2% —— **统计显著、无实用差异**。
- **训练数据量** 1k / 3k / 5k（固定 300 步）→ 1k 最差（同一批数据重复看 2.4 遍），据此定位"重复样本有害"；同时说明该设计下抽样混淆无法分离，**不外推数据量效应**。

**一条贯穿全项目的结论**：微调改变的是**行为和格式**，不是**知识容量**。LoRA 在这里只改 0.11%–0.44% 的参数，相当于在原书上贴便签——书里本来没有的那一页，贴多少便签也变不出来。

---

## 复现

**环境**

```bash
conda create -n p0 python=3.11
pip install torch==2.5.1 transformers peft bitsandbytes accelerate datasets
```

模型放在 `C:\models\Qwen2.5-0.5B-Instruct`（路径在脚本里以常量给出，按需修改）。

**A2 主线顺序**

```bash
python d28_dataset.py          # 数据清洗与 9:1 切分
python d29_sft_real.py         # QLoRA 微调（3000 条 / 300 步）
python d31_eval.py             # 留出集 PPL 评测
python d32_ablation.py 4       # r 消融（4 / 8 / 16）
python d34_build_domain_data.py --wan   # 自造领域数据（招聘 JD → schema JSON）
python d35_domain_sft.py train_wan.jsonl domain_v3
python d36_domain_eval.py      # 字段级自动评测
python d37_ood_eval.py         # 分布外评测
python d38_ood_b.py lora_out\domain_v3\final v3   # 独立终审集
python d39_ablation_scale.py 5000                 # 数据量消融
```

> **数据与 LoRA 权重不入库**（见 `.gitignore`：`data/`、`data_domain/`、`lora_out/`）。
> 所有实验数字都能在 `data/*.json` 与 `REPORT_A2_v1.md` 中找到出处。

---

## 命名规范

- `d<学习日>_<英文关键词>.py` —— 知识点演示与实验脚本
- `lc_<题号>_<中文题名>.py` —— LeetCode 题解（23 道）
- `ds_<结构名>.py` —— 数据结构手写实现

---

## 说明

这是**个人学习与实验仓库**，不是可安装的库。代码以可读性优先：关键行有中文注释，实验脚本保留完整参数以便复现。

如果只看一个文件，看 [`REPORT_A2_v1.md`](REPORT_A2_v1.md)。
