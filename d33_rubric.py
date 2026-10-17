# -*- coding: utf-8 -*-
"""d33：对比测试集 + 人工 rubric —— 补上 PPL 补不上的那一块（A2 第 4 节收尾）
跑法：cd C:\code\p0-python
      python d33_rubric.py

== 这个脚本要解决什么问题 ==
d31 算出来的 PPL 只能说明"像不像这份数据的风格"，它不检查答案对不对。
所以要另做一次评测：出 25 道题，让「原模型」和「微调后模型」各答一遍，
把两份答案并排放在一个文件里，由【人来打分】。

== 打分规则（记在 d33_testset.json 里，每题都有专属要点）==
  2 分 = 完全满足（内容对 + 格式和约束都对）
  1 分 = 部分满足（大体对，但有遗漏 / 没遵守某条约束 / 有冗余）
  0 分 = 不满足（答错、跑题、没遵守约束）
  满分 50。

== 题目的四类，以及预期 ==
  A 格式/风格(8 题) —— 微调擅长的，预期微调后 ≥ 原模型
  B 指令遵循(5 题) —— 约束类，预期微调后 ≥ 原模型
  C 能力    (7 题) —— 算术/推理，按 6.1 的结论，预期【两边都做不好、微调也补不上】
  D 中文语言(5 题) —— 预期两边接近

  ⚠️ 预期不是结论。**如果 A/B 类也没差别，那本身就是一个有力的结论**：
     它说明"PPL 下降"和"回答质量提升"确实是两件事，正好印证报告 6.2。

== 本文件里要你【看懂】的 ==
  只有一处：为什么两次生成都用 do_sample=False（贪心解码）—— 为了让结果可复现。
  其余全是 d31 见过的样板。

== 产物 ==
  data/rubric_result.md    ← 并排对比表，把它打开，人工往里填分数
  data/rubric_result.json  ← 原始答案，留档
"""
import os, gc, json, time                      # os 管路径 / gc 清显存 / json 读写 / time 计时
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL_PATH = r"C:\models\Qwen2.5-0.5B-Instruct"        # 基座模型
ADAPTER    = r"C:\code\p0-python\lora_out\real\final"    # d29 训出来的 LoRA 便签
DATA_DIR   = r"C:\code\p0-python\data"                   # 数据目录
TESTSET    = r"C:\code\p0-python\d33_testset.json"       # 测试集（25 题）

MAX_NEW_TOKENS = 96      # 每题最多生成多少个 token（够答完，又不至于太慢）

bnb = BitsAndBytesConfig(                                # 4bit 配置（同 d29/d31）
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True)


def build_model(adapter_dir=None):
    """adapter_dir=None → 裸模型；给了路径 → 贴上 LoRA 便签。"""
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tok.pad_token is None:                            # Qwen 有时没有 pad_token
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=bnb,
                                             device_map="auto")
    if adapter_dir is not None:
        m = PeftModel.from_pretrained(m, adapter_dir)     # 挂便签
    m.eval()                                             # 推理模式
    return m, tok


# ══════════════════════ 【看懂】生成回答 ══════════════════════
def ask(model, tok, question):
    """问一个问题，返回模型的回答（去掉提问部分）。"""
    # 把单个问题套进 chat 模板；add_generation_prompt=True 会加上"轮到你说了"的标记
    text = tok.apply_chat_template([{"role": "user", "content": question}],
                                   tokenize=False, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt").to(model.device)   # 转 id 并搬到 GPU

    with torch.no_grad():                                   # 不建计算图
        out = model.generate(
            **inp,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,       # ★ 贪心解码：每步都取概率最大的那个 token
                                   #   为什么不采样？因为要可复现 ——
                                   #   同样的输入必须得到同样的输出，否则两次评测
                                   #   的差异里会混进"随机性"这个噪声。
            pad_token_id=tok.pad_token_id,
        )
    # out[0] 是完整的 token 序列（提问 + 回答）；提问占了 inp["input_ids"].shape[1] 个
    # 所以从那个位置往后切片，才是模型自己生成的部分
    gen = out[0][inp["input_ids"].shape[1]:]
    return tok.decode(gen, skip_special_tokens=True).strip()   # 解回文字


def run_all(model, tok, items):
    """把 25 题全问一遍，返回 {题号: 回答} 的字典。"""
    ans = {}
    for it in items:                                     # 逐题
        t0 = time.time()
        ans[it["id"]] = ask(model, tok, it["q"])         # 生成回答
        print("  %-4s 用时 %.1f 秒" % (it["id"], time.time() - t0))
    return ans
# ═══════════════════════════════════════════════════════════


def main():
    items = json.load(open(TESTSET, encoding="utf-8"))   # 读 25 题
    print("测试集共 %d 题" % len(items))

    # ── ① 裸模型答题 ──
    print("\n[1/2] 裸模型（未微调）答题中...")
    model, tok = build_model(adapter_dir=None)
    ans_base = run_all(model, tok, items)
    del model                                            # 删掉腾显存
    gc.collect()
    torch.cuda.empty_cache()

    # ── ② 贴便签的模型答题 ──
    print("\n[2/2] LoRA 微调后模型答题中...")
    model, tok = build_model(adapter_dir=ADAPTER)
    ans_tuned = run_all(model, tok, items)

    # ── ③ 写成并排对比表（Markdown），空着两列分数等人填 ──
    lines = ["# d33 · 对比测试集评测结果（25 题）",
             "",
             "> 打分规则：**2 = 完全满足 / 1 = 部分满足 / 0 = 不满足**，满分 50。",
             "> 每题的具体评分要点见 `d33_testset.json`。填完两个分数列后，把总分填到最下面。",
             ""]
    cur_cat = None
    for it in items:
        if it["cat"] != cur_cat:                          # 换类别就插一个小标题
            cur_cat = it["cat"]
            lines += ["", "## %s（%s）" % (cur_cat, it["id"]), ""]
        lines += [
            "### %s" % it["id"],
            "",
            "**提问**：%s" % it["q"],
            "",
            "**评分要点**：%s" % it["rubric"],
            "",
            "| 模型 | 回答 | 分数 |",
            "|---|---|---|",
            "| 未微调 | %s |  |" % ans_base[it["id"]].replace("\n", "<br>").replace("|", "\\|"),
            "| 微调后 | %s |  |" % ans_tuned[it["id"]].replace("\n", "<br>").replace("|", "\\|"),
            "",
        ]
    lines += ["---", "", "## 得分汇总（填完上面后统计）", "",
              "| | 未微调 | 微调后 |", "|---|---|---|",
              "| A 格式/风格（满分 16） |  |  |",
              "| B 指令遵循（满分 10） |  |  |",
              "| C 能力（满分 14） |  |  |",
              "| D 中文语言（满分 10） |  |  |",
              "| **总分（满分 50）** |  |  |", ""]

    md_path = os.path.join(DATA_DIR, "rubric_result.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # ── ④ 同时存一份原始 json，留档 ──
    js_path = os.path.join(DATA_DIR, "rubric_result.json")
    with open(js_path, "w", encoding="utf-8") as f:
        json.dump({"base": ans_base, "tuned": ans_tuned}, f, ensure_ascii=False, indent=2)

    print("\n✅ 对比表已生成：" + md_path)
    print("✅ 原始答案已留档：" + js_path)
    print("👉 打开 rubric_result.md，对着「评分要点」给两列填 2 / 1 / 0")


if __name__ == "__main__":
    main()
