# -*- coding: utf-8 -*-
"""d36：领域任务的自动评测 —— 字段级准确率（A2 第 6 节收尾）
跑法：cd C:\code\p0-python
      python d36_domain_eval.py
（约 3~4 分钟，占满 GPU）

== 这一步比 d33（人工打分）好在哪 ==
d33 要人工给每题打 0/1/2，主观、慢、样本量上不去。
这次任务的答案是一个 JSON，**对错可以程序判定**：
  · parse_rate   —— 输出能不能被解析成 JSON（连格式都保不住 = 完全没学会 schema）
  · field_acc    —— 8 个字段逐个比对，算字段级准确率（看它哪里强、哪里弱）
  · exact_match  —— 8 个字段全对的比例（最严格的指标）
指标可自动算 → 样本量、可复现、跨模型可比，三个问题一起解决。

== 本文件里要你【看懂】的只有 2 处 ==
  ① extract_json()：怎么从一段可能带多余文字的模型输出里"抠"出 JSON
  ② compare()：怎么逐字段比对（为什么 skills 要用集合比、null 要单独判）
其余（加载模型 / 生成 / 写文件）都是 d31、d33 见过的样板。

== 产物 ==
  data_domain/eval_result.md    —— 人看的对比报告
  data_domain/eval_result.json  —— 原始结果留档
"""
import os, gc, json, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL_PATH = r"C:\models\Qwen2.5-0.5B-Instruct"
ADAPTER    = r"C:\code\p0-python\lora_out\domain\final"   # ★ d35 训出来的领域便签
DATA_DIR   = r"C:\code\p0-python\data_domain"
MAX_NEW_TOKENS = 160      # 一整段 JSON 大概 120~150 个 token

# 字段顺序 —— 和 d34 造数据时完全一致
FIELDS = ["company", "title", "city", "education", "experience_years",
          "skills", "salary_min_k", "salary_max_k"]

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)


def build_model(adapter_dir=None):
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(MODEL_PATH, quantization_config=bnb,
                                             device_map="auto")
    if adapter_dir is not None:
        m = PeftModel.from_pretrained(m, adapter_dir)
    m.eval()
    return m, tok


# ══════════════════════ 【看懂 ①】从输出里抠 JSON ══════════════════════
def extract_json(text):
    """模型可能输出 ```json {...} ``` 或加一句"好的，结果是："，这里只取 {...} 部分。

    做法：找【第一个左花括号】到【最后一个右花括号】之间的那段，再试 json.loads。
    返回：解析成功 → 字典；失败 → None。
    """
    i = text.find("{")          # 第一个 {
    j = text.rfind("}")         # 最后一个 }
    if i == -1 or j == -1 or j <= i:
        return None             # 没有大括号 → 一定不是 JSON
    try:
        return json.loads(text[i:j + 1])
    except Exception:           # 有大括号但不是合法 JSON（少引号、多逗号…）
        return None
# ═══════════════════════════════════════════════════════════════════


# ══════════════════════ 【看懂 ②】逐字段比对 ══════════════════════
def canon_list(x):
    """把「技能列表」规范化成可比较的形式。

    ⚠️ 教训（d37 第一次跑就崩了）：原先写的是 set(g) == set(p)。
       set() 要求每个元素可哈希，而模型可能在数组里塞 dict（例如
       {"skills": ["Go", {"name": "Rust"}]}）→ dict 不可哈希 → 直接 TypeError 崩掉。
       **打分函数崩了 = 什么都没测到。评测代码必须比被测模型更健壮。**

    做法：把每个元素先 json 序列化成字符串，再排序 —— 任何元素都能比，且顺序无关。
    """
    if not isinstance(x, list):
        return None            # 不是数组（比如模型给了一个字符串）→ 上层判为不相等
    return sorted(json.dumps(e, ensure_ascii=False, sort_keys=True) for e in x)


def compare(gold, pred):
    """比对一条预测和一个标准答案，返回 (对几个, 共几个, 错在哪些字段)。"""
    if pred is None:                       # 连 JSON 都解析不出来 → 全错
        return 0, len(FIELDS), list(FIELDS)
    ok, bad = 0, []
    for f in FIELDS:
        g = gold.get(f, None)
        p = pred.get(f, None)
        if f == "skills":
            # 技能是数组：顺序不算错，所以规范化后比（见 canon_list 的说明）。
            # 但 null 必须单独判 —— "没提技能"和"技能为空列表"不是一回事
            if g is None or p is None:
                same = (g is None and p is None)
            else:
                cg, cp = canon_list(g), canon_list(p)
                same = (cg is not None and cp is not None and cg == cp)
        else:
            # 其余字段严格要求值相等。
            # 注意：这是【严格】比对 —— 标准答案是数字 40，模型输出字符串 "40" 也算错。
            # 因为 schema 的意义就是"格式必须固定"，格式漂了就等于没学会。
            same = (g == p)
        if same:
            ok += 1
        else:
            bad.append(f)
    return ok, len(FIELDS), bad
# ═══════════════════════════════════════════════════════════════════


def ask(model, tok, prompt):
    """生成一条回答（贪心解码，保证可复现）。"""
    text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)
    inp = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=MAX_NEW_TOKENS,
                             do_sample=False, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def evaluate(model, tok, rows):
    """对评测集逐条生成 + 算分，返回逐条明细。"""
    detail = []
    for i, r in enumerate(rows):
        prompt = r["messages"][0]["content"]                     # 提问部分
        gold = json.loads(r["messages"][1]["content"])           # 标准答案
        pred_text = ask(model, tok, prompt)                      # 模型输出
        pred = extract_json(pred_text)                           # 抠 JSON
        ok, tot, bad = compare(gold, pred)                       # 比字段
        detail.append({"idx": i, "gold": gold, "pred_text": pred_text,
                       "pred": pred, "ok": ok, "total": tot, "bad": bad})
        if (i + 1) % 20 == 0:
            print("  已评 %d/%d 条" % (i + 1, len(rows)))
    return detail


def summarize(detail):
    """把逐条明细汇总成几个指标。"""
    n = len(detail)
    parse = sum(1 for d in detail if d["pred"] is not None) / n          # JSON 可解析率
    exact = sum(1 for d in detail if d["ok"] == d["total"]) / n          # 全字段正确率
    # 字段级准确率：把所有样本的所有字段摊平了算
    fld_ok = {f: 0 for f in FIELDS}
    for d in detail:
        for f in FIELDS:
            if f not in d["bad"]:
                fld_ok[f] += 1
    fld_acc = {f: fld_ok[f] / n for f in FIELDS}
    avg = sum(d["ok"] / d["total"] for d in detail) / n                  # 平均字段正确率
    return {"n": n, "parse_rate": parse, "exact_match": exact,
            "field_acc": fld_acc, "avg_field_acc": avg}


def main():
    rows = [json.loads(l) for l in open(os.path.join(DATA_DIR, "eval.jsonl"), encoding="utf-8")]
    print("评测集 = %d 条" % len(rows))

    # ── ① 裸模型 ──
    print("\n[1/2] 裸模型（未微调）...")
    m, t = build_model(None)
    d_base = evaluate(m, t, rows)
    s_base = summarize(d_base)
    del m; gc.collect(); torch.cuda.empty_cache()

    # ── ② 贴领域便签的模型 ──
    print("\n[2/2] 领域 LoRA 微调后...")
    m, t = build_model(ADAPTER)
    d_tuned = evaluate(m, t, rows)
    s_tuned = summarize(d_tuned)

    # ── ③ 打印对比 ──
    print("\n" + "=" * 62)
    print("%-22s %12s %12s" % ("指标", "未微调", "微调后"))
    print("-" * 62)
    print("%-22s %11.1f%% %11.1f%%" % ("JSON 可解析率", s_base["parse_rate"] * 100,
                                       s_tuned["parse_rate"] * 100))
    print("%-22s %11.1f%% %11.1f%%" % ("平均字段正确率", s_base["avg_field_acc"] * 100,
                                       s_tuned["avg_field_acc"] * 100))
    print("%-22s %11.1f%% %11.1f%%" % ("全字段完全正确率", s_base["exact_match"] * 100,
                                       s_tuned["exact_match"] * 100))
    print("-" * 62)
    for f in FIELDS:
        print("%-22s %11.1f%% %11.1f%%" % ("  字段 " + f, s_base["field_acc"][f] * 100,
                                           s_tuned["field_acc"][f] * 100))
    print("=" * 62)

    # ── ④ 写文件 ──
    out = {"base": s_base, "tuned": s_tuned}
    js = os.path.join(DATA_DIR, "eval_result.json")
    with open(js, "w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False, indent=2)

    md = ["# d36 · 领域任务自动评测（招聘 JD → JSON 抽取）", "",
          "评测集 %d 条（复用 `data_domain/eval.jsonl`，与训练集零重叠）。" % len(rows), "",
          "| 指标 | 未微调 | 微调后 |", "|---|---|---|",
          "| JSON 可解析率 | %.1f%% | %.1f%% |" % (s_base["parse_rate"]*100, s_tuned["parse_rate"]*100),
          "| 平均字段正确率 | %.1f%% | %.1f%% |" % (s_base["avg_field_acc"]*100, s_tuned["avg_field_acc"]*100),
          "| 全字段完全正确率 | %.1f%% | %.1f%% |" % (s_base["exact_match"]*100, s_tuned["exact_match"]*100),
          "", "## 分字段准确率", "", "| 字段 | 未微调 | 微调后 |", "|---|---|---|"]
    for f in FIELDS:
        md.append("| `%s` | %.1f%% | %.1f%% |" % (f, s_base["field_acc"][f]*100, s_tuned["field_acc"][f]*100))
    md += ["", "## 样例（前 5 条）", ""]
    for d in d_tuned[:5]:
        md += ["**提问**：%s" % rows[d["idx"]]["messages"][0]["content"][-140:], "",
               "- 标准答案：`%s`" % json.dumps(d["gold"], ensure_ascii=False),
               "- 微调后输出：`%s`" % d["pred_text"].replace("\n", " ")[:220],
               "- 得分：%d/%d%s" % (d["ok"], d["total"],
                                    ("  错在 " + ", ".join(d["bad"])) if d["bad"] else ""), ""]
    mp = os.path.join(DATA_DIR, "eval_result.md")
    with open(mp, "w", encoding="utf-8") as fp:
        fp.write("\n".join(md))

    print("\n✅ 已存到", mp)
    print("✅ 已存到", js)


if __name__ == "__main__":
    main()
