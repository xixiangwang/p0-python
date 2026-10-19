# -*- coding: utf-8 -*-
"""d38：OOD-B —— 独立终审测试集（只准跑一次）
跑法：cd C:\code\p0-python
      python d38_ood_b.py
（约 3~4 分钟，占满 GPU）

== 为什么还要再造一份 ==
OOD-A（d37 的 110 条）已经被"用过了"：
    跑 OOD-A → 看到 10 个错误 → 归因 → 改训练数据 → 回到 OOD-A 复测 → 100%
这就等于 **同一个测试集既用来找问题、又用来验证修复**，
在机器学习里叫 test set reuse，本质是"对 benchmark 过拟合"。
所以那个 100% 不能作为最终结论。

== 规矩（定死，不许破）==
**OOD-B 造出来之后只跑一次，报那个数，不许再根据结果回去调训练数据。**
一旦调了，OOD-B 也就废了 —— 那时你就没有独立测试集了。

== 这份测的是什么（都是训练集和 OOD-A 里【没有】的东西）==
① negotiable  —— 薪资**面议 / 另议 / 面谈** → 标准答案是 null（不是 0，也不是猜一个数）
② register    —— **公告体 / 口语体**（不带"招聘"两字的公文腔、微信聊天腔）
③ codeswitch  —— **中英混杂**（技能和描述夹英文）
④ unit        —— 薪资用「**万**」表达（"3万-4.5万"），而 schema 的单位是「千元」

== 我（zt）的预判，先写在这里，事后好对账 ==
· ① negotiable 会明显掉 —— 因为"该给 null"这种语义类型在训练集里完全不存在
· ④ unit 也会掉 —— 单位换算训练集没有
· ② register 大概中等
· ③ codeswitch 应该还行（技能名本来就是英文）
**如果①和④没掉，才说明模型真的学到了"读数"而不是"读位置"。**

== 产物 ==
  data_domain/ood_b.jsonl                                 测试集
  data_domain/ood_b_result.md / .json                      结果
"""
import os, sys, json, gc, random

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from d36_domain_eval import (extract_json, compare, build_model,
                             evaluate, summarize, FIELDS)

MODEL_PATH = r"C:\models\Qwen2.5-0.5B-Instruct"
ADAPTER    = sys.argv[1] if len(sys.argv) > 1 else r"C:\code\p0-python\lora_out\domain_v2\final"
TAG        = sys.argv[2] if len(sys.argv) > 2 else "norm"     # 输出文件名后缀
DATA_DIR   = r"C:\code\p0-python\data_domain"
INSTRUCTION = ("从下面的招聘信息中抽取字段并输出 JSON，字段：company, title, city, "
               "education, experience_years, skills(数组), salary_min_k, salary_max_k。"
               "未提到的字段填 null。")   # ★ 与训练时一字不差

# ══════ OOD-B 专用取值池：和 d34（训练）与 d37（OOD-A）【完全不重叠】══════
B_COMPANIES = ["中芯国际", "潍柴动力", "立讯精密", "药明康德", "海康威视",
               "大疆创新", "寒武纪", "用友网络", "金蝶软件", "汇川技术"]
B_CITIES = ["佛山", "东莞", "无锡", "常州", "济南", "福州", "昆明", "贵阳"]
B_TITLES = ["大模型训练工程师", "推理引擎开发工程师", "对话系统算法工程师",
            "知识图谱工程师", "算法研究员（NLP）"]
B_SKILLS = ["CUDA", "Triton", "Megatron", "DeepSpeed", "Ray", "Milvus", "Faiss",
            "Qwen", "Llama", "SFT", "DPO", "RLHF"]


def gold(c, t, ci, edu, exp, sk, smin, smax):
    return {"company": c, "title": t, "city": ci, "education": edu,
            "experience_years": exp, "skills": sk,
            "salary_min_k": smin, "salary_max_k": smax}


def wrap(jd, g):
    return {"messages": [
        {"role": "user", "content": INSTRUCTION + "\n\n招聘信息：" + jd},
        {"role": "assistant", "content": json.dumps(g, ensure_ascii=False)},
    ]}


def pick(rng):
    c = rng.choice(B_COMPANIES)
    t = rng.choice(B_TITLES)
    ci = rng.choice(B_CITIES)
    edu = rng.choice(["本科", "硕士", "博士"])
    exp = rng.choice([1, 2, 3, 4, 6])
    sk = rng.sample(B_SKILLS, rng.randint(2, 4))
    smin = rng.choice([20, 25, 30, 35, 40, 45, 50])
    smax = smin + rng.choice([10, 15, 20, 25])
    return c, t, ci, edu, exp, sk, smin, smax


EDU_TXT = {"本科": "本科及以上", "硕士": "硕士及以上", "博士": "博士"}


# ══════════════════════ 【看懂 ①】薪资面议 → null ══════════════════════
# 关键：训练集里"该输出 null"只有一个原因 —— 字段压根没写。
# 这里换一种：字段【写了】，但写的是"面议" —— 依然没有任何数值可抽。
# 模型必须明白"面议"≠一个数字，也不能编。
def make_negotiable(rng):
    c, t, ci, edu, exp, sk, smin, smax = pick(rng)
    phrase = rng.choice(["薪资面议，具体待遇根据能力与经验确定。",
                         "薪酬另议。",
                         "具体薪资面谈，公司提供有竞争力的package。",
                         "待遇按公司薪酬制度执行，不做统一披露。"])
    jd = ("%s现招聘%s。工作城市：%s。任职要求：%s，%d 年相关工作经验，"
          "熟练掌握%s。%s" % (c, t, ci, EDU_TXT[edu], exp, "、".join(sk), phrase))
    g = gold(c, t, ci, edu, exp, sk, None, None)     # ★ 薪资两个字段必须是 null
    return wrap(jd, g)


# ══════════════════════ 【看懂 ②】公告体 / 口语体 ══════════════════════
# 训练集和 OOD-A 的句式都是"新闻报道式的招聘广告"。
# 这里换成两种完全不同的语域：公文腔、微信聊天腔。
def make_register(rng):
    c, t, ci, edu, exp, sk, smin, smax = pick(rng)
    sks = "、".join(sk)
    if rng.random() < 0.5:
        # 公文腔：长句、无"招聘"两字、信息埋在从句里
        jd = ("经部门研究并报公司批准，%s决定面向社会公开选聘%s若干名。"
              "录用人员工作地点为%s，须具备%s学历及%d 年以上相关岗位从业经历，"
              "并熟练掌握%s。薪酬按公司现行制度执行，月薪区间为 %d 至 %d 千元。"
              % (c, t, ci, EDU_TXT[edu], exp, sks, smin, smax))
    else:
        # 口语腔：短句、聊天语气、数字和标点不规范
        jd = ("哈喽，我们%s这边在招个%s，base %s，"
              "要%s，%d年经验，会用%s，"
              "给 %d 到 %d K，有兴趣的可以聊一下"
              % (c, t, ci, EDU_TXT[edu], exp, sks, smin, smax))
    return wrap(jd, gold(c, t, ci, edu, exp, sk, smin, smax))


# ══════════════════════ 【看懂 ③】中英混杂 ══════════════════════
# 技能和描述夹英文（公司名/城市/学历仍保持中文，避免 gold 有歧义）
def make_codeswitch(rng):
    c, t, ci, edu, exp, sk, smin, smax = pick(rng)
    sks = " / ".join(sk)
    jd = ("%s 正在招聘 %s。Base in %s。"
          "Requirement：%s及以上，%d 年 experience。"
          "Must be familiar with %s。"
          "Package：%d-%dK per month。"
          % (c, t, ci, EDU_TXT[edu], exp, sks, smin, smax))
    return wrap(jd, gold(c, t, ci, edu, exp, sk, smin, smax))


# ══════════════════════ 【看懂 ④】薪资用「万」表达 ══════════════════════
# schema 的单位是「千元」，而文本写「3万」「4.5万」。
# 必须做单位换算：3万 = 30（千元）。训练集里全是"30-45K"，没有这种写法。
def _wan(x):
    s = "%.1f" % (x / 10.0)
    return s[:-2] if s.endswith(".0") else s


def make_unit(rng):
    c, t, ci, edu, exp, sk, smin, smax = pick(rng)
    jd = ("%s招聘%s，工作地点：%s。%s，%d 年工作经验，熟悉%s。"
          "月薪 %s万 - %s万元。"
          % (c, t, ci, EDU_TXT[edu], exp, "、".join(sk), _wan(smin), _wan(smax)))
    return wrap(jd, gold(c, t, ci, edu, exp, sk, smin, smax))
# ═══════════════════════════════════════════════════════════════════


def build():
    rng = random.Random(20261017)      # ★ 和 d34（42/43）、d37（777）都不同
    items = []
    for _ in range(25):
        items.append((make_negotiable(rng), "negotiable"))
    for _ in range(25):
        items.append((make_register(rng), "register"))
    for _ in range(25):
        items.append((make_codeswitch(rng), "codeswitch"))
    for _ in range(25):
        items.append((make_unit(rng), "unit"))
    # 查重
    seen, uniq = set(), []
    for m, dim in items:
        k = m["messages"][0]["content"]
        if k in seen:
            continue
        seen.add(k)
        uniq.append({"messages": m["messages"], "dim": dim})
    return uniq


def main():
    ood = build()
    from collections import Counter
    print("OOD-B = %d 条 | 维度 %s" % (len(ood), dict(Counter(x["dim"] for x in ood))))

    path = os.path.join(DATA_DIR, "ood_b.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for x in ood:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print("已存到", path)

    # 泄漏自查：与 train / eval / ood-A 都不能重叠
    def keys(fp):
        return set(json.dumps(json.loads(l)["messages"], ensure_ascii=False, sort_keys=True)
                   for l in open(fp, encoding="utf-8") if l.strip())
    mine = set(json.dumps(x["messages"], ensure_ascii=False, sort_keys=True) for x in ood)
    for name in ["train.jsonl", "eval.jsonl", "ood.jsonl"]:
        fp = os.path.join(DATA_DIR, name)
        if os.path.exists(fp):
            print("  与 %-12s 重叠 = %d" % (name, len(mine & keys(fp))))

    # ── 只跑微调后的模型（基座在 OOD-A 上已经测过，这里只关心终审）──
    print("\n加载模型并评测（%d 条，约 2~3 分钟）..." % len(ood))
    m, t = build_model(ADAPTER)
    detail = evaluate(m, t, ood)
    s = summarize(detail)

    dims = ["negotiable", "register", "codeswitch", "unit"]
    dim_cn = {"negotiable": "① 薪资面议 → null",
              "register": "② 公告体 / 口语体",
              "codeswitch": "③ 中英混杂",
              "unit": "④ 薪资用「万」表达"}

    def by_dim(dim):
        sub = [d for d in detail if ood[d["idx"]]["dim"] == dim]
        n = len(sub)
        return {"n": n,
                "parse": sum(1 for d in sub if d["pred"] is not None) / n,
                "exact": sum(1 for d in sub if d["ok"] == d["total"]) / n,
                "avg": sum(d["ok"] / d["total"] for d in sub) / n}

    print("\n" + "=" * 70)
    print("OOD-B（独立终审集，只跑这一次）")
    print("-" * 70)
    print("%-30s %14s" % ("指标", "微调后 v2"))
    print("  JSON 可解析率            %11.1f%%" % (s["parse_rate"] * 100))
    print("  平均字段正确率            %11.1f%%" % (s["avg_field_acc"] * 100))
    print("  全字段完全正确率          %11.1f%%" % (s["exact_match"] * 100))
    print("\n分维度：")
    dm_stats = {}
    for dm in dims:
        a = by_dim(dm)
        dm_stats[dm] = a
        print("  %-28s %8.1f%%  (n=%d)" % (dim_cn[dm], a["exact"] * 100, a["n"]))
    print("\n分字段：")
    for f in FIELDS:
        print("  %-24s %8.1f%%" % (f, s["field_acc"][f] * 100))
    print("=" * 70)

    out = {"overall": s, "by_dim": dm_stats}
    # ⚠️ 文件名带 _norm：不覆盖第一轮的严格口径结果（ood_b_result.json，50.0%）
    with open(os.path.join(DATA_DIR, "ood_b_result_%s.json" % TAG), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    md = ["# d38 · OOD-B 独立终审结果（规范化重算）", "",
          "> **基线**：第一轮用字符级严格比对，全字段完全正确率 **50.0%**（见 `ood_b_result.json`）。",
          "> **本文件**：修正「全角/半角括号、空格」这类**显示层差异**后重新计分。",
          "> 语义上的真错（如 '硕士' vs '硕士及以上'、40 vs '40'）依然判错。", "",
          "| 指标 | 修正后 |", "|---|---|",
          "| JSON 可解析率 | %.1f%% |" % (s["parse_rate"] * 100),
          "| 平均字段正确率 | %.1f%% |" % (s["avg_field_acc"] * 100),
          "| 全字段完全正确率 | %.1f%% |" % (s["exact_match"] * 100),
          "", "| 维度 | 全对率 | 条数 |", "|---|---|---|"]
    for dm in dims:
        md.append("| %s | %.1f%% | %d |" % (dim_cn[dm], dm_stats[dm]["exact"] * 100, dm_stats[dm]["n"]))
    md += ["", "## 出错样例", ""]
    shown = 0
    for d in detail:
        if d["bad"] and shown < 12:
            shown += 1
            md += ["**【%s】** %s" % (ood[d["idx"]]["dim"], ood[d["idx"]]["messages"][0]["content"][-160:]),
                   "", "- 标准答案：`%s`" % json.dumps(d["gold"], ensure_ascii=False),
                   "- 模型输出：`%s`" % d["pred_text"].replace("\n", " ")[:220],
                   "- 错在：%s" % ", ".join(d["bad"]), ""]
    mp = os.path.join(DATA_DIR, "ood_b_result_%s.md" % TAG)
    with open(mp, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("\n✅ 已存到", mp)


if __name__ == "__main__":
    main()
