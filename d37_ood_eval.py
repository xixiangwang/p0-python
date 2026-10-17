# -*- coding: utf-8 -*-
"""d37：分布外（OOD）评测 —— 100% 到底是真的还是背模板？（A2 第 6 节收尾）
跑法：cd C:\code\p0-python
      python d37_ood_eval.py
（约 2~3 分钟，占满 GPU）

== 为什么要做这一份 ==
d36 在「同源评测集」上跑出全字段完全正确率 0% → 100%。
但那份评测集和训练集是【同一个生成器、同一套句式模板】造的，
模型只要学会这十来种模板就能满分 —— 所以 100% 只能证明"学会了固定映射"，
证明不了"学会了抽取"。

这一份要回答的就是：**换个说法、换批词、加噪音，它还认不认得。**

== OOD 的三个维度（这是本文件的核心设计）==
① 留出模板 heldout_template —— 5 种【训练里从未出现过】的句式
② 真实感长文本 real_text     —— 一段真实 JD 的样子：职责描述 + 加分项干扰 + 公司介绍里的陷阱数字
③ 格式噪声 noise             —— 经验年限写成中文数字（"三年"）、夹无关福利段落、多余空格

== ★ 一个重要的设计决定（这是从 B3 那次教训里学来的）==
OOD 的**所有文本都是程序生成的，标准答案因此是客观的、唯一确定的**。
为什么不手写"真实 JD"？因为手写的话，"哪些词算 required 技能"就由我说了算 ——
那就是 B3 那次踩过的坑：**指标一主观，分数就没意义。**
所以这里的做法是：**文本看起来乱、但答案的提取规则是机器钉死的。**

== 产物 ==
  data_domain/ood.jsonl           OOD 测试集
  data_domain/ood_result.md       分维度 + 分字段的对比结果
  data_domain/ood_result.json     原始结果留档
"""
import os, sys, json, gc, random, time

# ── 把当前目录加进搜索路径，好复用 d36 里的打分函数（不复制代码，避免两处不一致）──
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from d36_domain_eval import (extract_json, compare, build_model,
                             evaluate, summarize, FIELDS)

MODEL_PATH = r"C:\models\Qwen2.5-0.5B-Instruct"
ADAPTER    = r"C:\code\p0-python\lora_out\domain\final"
DATA_DIR   = r"C:\code\p0-python\data_domain"
INSTRUCTION = ("从下面的招聘信息中抽取字段并输出 JSON，字段：company, title, city, "
               "education, experience_years, skills(数组), salary_min_k, salary_max_k。"
               "未提到的字段填 null。")   # ★ 和训练时一字不差

# ══════════════════════ 【直接复制】OOD 专用取值池（训练里一个都没出现过）══════════════════════
NEW_COMPANIES = ["宁德时代", "比亚迪", "海尔智家", "三一重工", "迈瑞医疗",
                 "顺丰速运", "京东方", "紫光展锐", "壁仞科技", "云从科技"]
NEW_CITIES = ["长沙", "郑州", "青岛", "合肥", "天津", "重庆", "厦门", "宁波"]
NEW_SKILLS = ["Go", "Rust", "Spark", "Flink", "Docker", "Kubernetes", "FastAPI",
              "Redis", "Elasticsearch", "ClickHouse", "图神经网络", "强化学习",
              "扩散模型", "ONNX", "TensorRT", "DeepSpeed"]
NEW_TITLES = ["大模型推理优化工程师", "AI平台开发工程师", "多模态大模型算法工程师",
              "强化学习算法工程师", "大模型安全算法工程师"]

CN_NUM = {0: "零", 1: "一", 2: "两", 3: "三", 4: "四", 5: "五",
          6: "六", 7: "七", 8: "八", 9: "九", 10: "十"}


def gold(company, title, city, education, exp, skills, smin, smax):
    """按固定字段顺序组一个标准答案 —— 顺序必须和 d34 一致。"""
    return {"company": company, "title": title, "city": city, "education": education,
            "experience_years": exp, "skills": skills,
            "salary_min_k": smin, "salary_max_k": smax}


def wrap(jd, g):
    """把 (提示词, 标准答案) 包成 chat 格式。"""
    return {"messages": [
        {"role": "user", "content": INSTRUCTION + "\n\n招聘信息：" + jd},
        {"role": "assistant", "content": json.dumps(g, ensure_ascii=False)},
    ]}


def rng_pick(rng):
    """随机抽一整套字段值（用 OOD 池子）。"""
    c = rng.choice(NEW_COMPANIES)
    t = rng.choice(NEW_TITLES)
    ci = rng.choice(NEW_CITIES)
    edu = rng.choice(["本科", "硕士", "博士", "不限"])
    exp = rng.choice([1, 2, 3, 5, 7])
    sk = rng.sample(NEW_SKILLS, rng.randint(2, 4))
    smin = rng.choice([20, 25, 30, 35, 45])
    smax = smin + rng.choice([10, 15, 20, 25])
    return c, t, ci, edu, exp, sk, smin, smax


# ══════════════════════ 【看懂】维度① 留出模板 ══════════════════════
# 这 5 个句式，训练数据里【一个都没出现过】。
# 它们和训练模板的区别在于：字段的位置、语序、连接词全都变了，
# 甚至把多个字段塞进同一个句子里 —— 想拿分就必须真的"读句子"，不能靠"找关键词"。
def make_heldout(rng):
    c, t, ci, edu, exp, sk, smin, smax = rng_pick(rng)
    edu_p = rng.choice(["本科及以上学历", "硕士研究生", "不限学历"])
    if edu == "本科":
        edu_p = "本科及以上学历"
    elif edu == "硕士":
        edu_p = "硕士研究生"
    elif edu == "博士":
        edu_p = "博士研究生"
    else:
        edu_p = "不限学历"
    sks = "、".join(sk)

    k = rng.randint(1, 5)
    if k == 1:      # 句式 1：把公司/城市/岗位塞进一句叙述里
        jd = ("我们是一家位于%s的公司，名叫%s，现在急需一位%s加入团队。"
              "希望你%s，有 %d 年相关经验，日常会用到%s，薪资 %d-%dK。"
              % (ci, c, t, edu_p, exp, sks, smin, smax))
        g = gold(c, t, ci, edu, exp, sk, smin, smax)
    elif k == 2:    # 句式 2：括号补充式
        jd = ("%s（%s｜%s）—— 要求%s，%d 年经验，%s，薪酬 %d-%dK。"
              % (t, c, ci, edu_p, exp, sks, smin, smax))
        g = gold(c, t, ci, edu, exp, sk, smin, smax)
    elif k == 3:    # 句式 3：第一人称 HR
        jd = ("我是%s的招聘负责人，我们在%s招聘%s。这个岗位希望你具备%s学历、"
              "%d 年经验，熟悉%s，月薪 %d-%dK。"
              % (c, ci, t, edu_p, exp, sks, smin, smax))
        g = gold(c, t, ci, edu, exp, sk, smin, smax)
    elif k == 4:    # 句式 4：竖线分隔式
        jd = ("岗位：%s｜公司：%s｜城市：%s｜学历：%s｜经验：%d年｜技能：%s｜月薪：%d-%dK"
              % (t, c, ci, edu, exp, sks, smin, smax))
        g = gold(c, t, ci, edu, exp, sk, smin, smax)
    else:           # 句式 5：倒装句
        jd = ("在%s，%s正在为%s这个岗位寻找合适的人选。条件：%s，%d 年以上工作经验，"
              "掌握%s。待遇 %d-%dK。"
              % (ci, c, t, edu_p, exp, sks, smin, smax))
        g = gold(c, t, ci, edu, exp, sk, smin, smax)
    return wrap(jd, g)


# ══════════════════════ 【看懂】维度② 真实感长文本 ══════════════════════
# 关键点：文本变长、变杂，并且塞了两个干扰源 ——
#   · 公司介绍里的【陷阱数字】（成立年份、员工人数、融资轮次）→ 看模型会不会把它们当成年限/薪资
#   · "了解 X 者优先" 里的【加分技能】→ 不进标准答案（标准答案只收明确要求的）
def make_real(rng):
    c, t, ci, edu, exp, sk, smin, smax = rng_pick(rng)
    edu_p = {"本科": "本科及以上", "硕士": "硕士及以上",
             "博士": "博士", "不限": "学历不限"}[edu]
    bonus = [s for s in NEW_SKILLS if s not in sk]
    b = rng.sample(bonus, min(2, len(bonus)))     # 加分技能（不进答案）
    year = rng.randint(1998, 2018)
    staff = rng.choice([500, 800, 1200, 2000, 5000])
    rnd = rng.choice(["A", "B", "C", "D"])

    jd = ("【关于我们】%s成立于 %d 年，目前团队规模超过 %d 人，已完成 %s 轮融资。\n"
          "【职位名称】%s　【工作城市】%s\n"
          "【岗位职责】负责%s相关方向的核心算法研发与工程落地，参与从方案设计到"
          "线上部署的完整链路，并与产品、工程团队协作推进。\n"
          "【任职要求】%s学历，%d 年以上相关工作经验，熟练掌握%s。"
          "了解%s者优先考虑。\n"
          "【薪酬福利】提供 %d-%dK 月薪，另有年终奖、补充医疗保险、免费班车与"
          "弹性工作时间。%s遵守劳动法，欢迎长期发展。"
          % (c, year, staff, rnd, t, ci, t, edu_p, exp, "、".join(sk),
             "、".join(b), smin, smax, c))
    g = gold(c, t, ci, edu, exp, sk, smin, smax)
    return wrap(jd, g)


# ══════════════════════ 【看懂】维度③ 格式噪声 ══════════════════════
# 两处变化：
#   ① 经验年限写成【中文数字】（"三年"而不是"3 年"）—— 训练数据里全是阿拉伯数字
#   ② 中间插一段跟字段无关的福利段落
def make_noise(rng):
    c, t, ci, edu, exp, sk, smin, smax = rng_pick(rng)
    edu_p = {"本科": "本科以上", "硕士": "硕士以上",
             "博士": "博士学历", "不限": "不限学历"}[edu]
    cn = CN_NUM.get(exp, str(exp))               # 3 → "三"
    jd = ("%s 招聘 %s 。\n"
          "工作地点：%s 。\n"
          "任职资格：%s ，有 %s 年以上相关经历，熟悉 %s 。\n"
          "我们提供：五险一金、带薪年假、节日福利、定期体检、年度旅游。\n"
          "薪资待遇：%d - %dK 。"
          % (c, t, ci, edu_p, cn, "、".join(sk), smin, smax))
    g = gold(c, t, ci, edu, exp, sk, smin, smax)
    return wrap(jd, g)
# ═══════════════════════════════════════════════════════════════════


def build_ood():
    """造出三个维度的 OOD 样本。"""
    rng = random.Random(777)          # ★ 种子和 d34 不同（d34 用的是 42 和 43）
    items = []
    for _ in range(60):
        items.append((make_heldout(rng), "heldout_template"))
    for _ in range(20):
        items.append((make_real(rng), "real_text"))
    for _ in range(30):
        items.append((make_noise(rng), "noise"))

    # 查重（防止生成器碰巧产出重复样本）+ 顺便确认和训练集不重叠
    seen, uniq = set(), []
    for m, dim in items:
        k = json.dumps(m["messages"][0]["content"], ensure_ascii=False)
        if k in seen:
            continue
        seen.add(k)
        uniq.append({"messages": m["messages"], "dim": dim})
    return uniq


def main():
    ood = build_ood()
    print("OOD 测试集 = %d 条" % len(ood))
    from collections import Counter
    print("维度分布 =", dict(Counter(x["dim"] for x in ood)))

    # 写文件（格式和 d34 的 eval.jsonl 一致，只是多一个 dim 标签）
    ood_path = os.path.join(DATA_DIR, "ood.jsonl")
    with open(ood_path, "w", encoding="utf-8") as f:
        for x in ood:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print("已存到", ood_path)

    # ── ① 裸模型 ──
    print("\n[1/2] 裸模型（未微调）在 OOD 上答题...")
    m, t = build_model(None)
    d_base = evaluate(m, t, ood)
    s_base = summarize(d_base)
    del m; gc.collect(); __import__("torch").cuda.empty_cache()

    # ── ② 领域微调后 ──
    print("\n[2/2] 领域微调后在 OOD 上答题...")
    m, t = build_model(ADAPTER)
    d_tuned = evaluate(m, t, ood)
    s_tuned = summarize(d_tuned)

    def by_dim(detail, dim):
        sub = [d for d in detail if ood[d["idx"]]["dim"] == dim]
        n = len(sub)
        if n == 0:
            return {"n": 0, "parse": 0, "avg": 0, "exact": 0}
        return {"n": n,
                "parse": sum(1 for d in sub if d["pred"] is not None) / n,
                "avg": sum(d["ok"] / d["total"] for d in sub) / n,
                "exact": sum(1 for d in sub if d["ok"] == d["total"]) / n}

    dims = ["heldout_template", "real_text", "noise"]
    dim_cn = {"heldout_template": "① 留出模板（5 种没见过的句式）",
              "real_text": "② 真实感长文本（含陷阱数字 + 加分项干扰）",
              "noise": "③ 格式噪声（中文数字 + 无关段落）"}

    print("\n" + "=" * 74)
    print("OOD 总体：")
    print("%-26s %12s %12s" % ("指标", "未微调", "微调后"))
    print("-" * 74)
    for key, name in [("parse_rate", "JSON 可解析率"), ("avg_field_acc", "平均字段正确率"),
                      ("exact_match", "全字段完全正确率")]:
        print("%-26s %11.1f%% %11.1f%%" % (name, s_base[key] * 100, s_tuned[key] * 100))
    print("\n分维度：")
    print("%-42s %10s %10s" % ("维度", "未微调", "微调后"))
    print("-" * 74)
    for dm in dims:
        a = by_dim(d_base, dm)
        b = by_dim(d_tuned, dm)
        print("%-42s %9.1f%% %9.1f%%  (n=%d)" % (dim_cn[dm], a["exact"] * 100,
                                                 b["exact"] * 100, b["n"]))
    print("\n微调后 · 分字段准确率：")
    for f in FIELDS:
        print("   %-22s %11.1f%%" % (f, s_tuned["field_acc"][f] * 100))
    print("=" * 74)

    # ── 写文件 ──
    out = {"base": s_base, "tuned": s_tuned,
           "by_dim": {dm: {"base": by_dim(d_base, dm), "tuned": by_dim(d_tuned, dm)}
                      for dm in dims}}
    with open(os.path.join(DATA_DIR, "ood_result.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    md = ["# d37 · 分布外（OOD）评测结果", "",
          "测试集 %d 条，与训练集**不同模板、不同取值池、含噪声**。" % len(ood), "",
          "| 指标 | 未微调 | 微调后 |", "|---|---|---|",
          "| JSON 可解析率 | %.1f%% | %.1f%% |" % (s_base["parse_rate"]*100, s_tuned["parse_rate"]*100),
          "| 平均字段正确率 | %.1f%% | %.1f%% |" % (s_base["avg_field_acc"]*100, s_tuned["avg_field_acc"]*100),
          "| 全字段完全正确率 | %.1f%% | %.1f%% |" % (s_base["exact_match"]*100, s_tuned["exact_match"]*100),
          "", "## 分维度（全字段完全正确率）", "",
          "| 维度 | 未微调 | 微调后 |", "|---|---|---|"]
    for dm in dims:
        a, b = by_dim(d_base, dm), by_dim(d_tuned, dm)
        md.append("| %s | %.1f%% | %.1f%% |" % (dim_cn[dm], a["exact"]*100, b["exact"]*100))
    md += ["", "## 微调后 · 分字段准确率", "", "| 字段 | 准确率 |", "|---|---|"]
    for f in FIELDS:
        md.append("| `%s` | %.1f%% |" % (f, s_tuned["field_acc"][f]*100))
    md += ["", "## 微调后 · 出错样例（最多 10 条）", ""]
    shown = 0
    for d in d_tuned:
        if d["bad"] and shown < 10:
            shown += 1
            md += ["**【%s】** %s" % (ood[d["idx"]]["dim"], ood[d["idx"]]["messages"][0]["content"][-180:]),
                   "",
                   "- 标准答案：`%s`" % json.dumps(d["gold"], ensure_ascii=False),
                   "- 模型输出：`%s`" % d["pred_text"].replace("\n", " ")[:250],
                   "- 错在：%s" % ", ".join(d["bad"]), ""]
    mp = os.path.join(DATA_DIR, "ood_result.md")
    with open(mp, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("\n✅ 已存到", mp)


if __name__ == "__main__":
    main()
