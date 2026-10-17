# -*- coding: utf-8 -*-
"""d34：自造领域数据 —— 招聘 JD → 结构化 JSON（A2 剩余清单第 ② 项）
跑法：cd C:\code\p0-python
      python d34_build_domain_data.py
（纯 CPU，几秒钟跑完，不碰 GPU）

== 为什么要做这一步 ==
现在的报告里，微调是「打平」的（人工评分 34 : 33）。
原因很清楚：alpaca-zh 教的是通用对话风格，而通用风格 Qwen-Instruct 本来就会。
要让微调真正有价值，必须换一个【基座一定做不稳、而数据能教明白】的任务。

== 这个任务为什么选它 ==
把自由文本转成「固定 schema 的 JSON」——
  · 它是【格式/行为】类任务 → 正好是微调擅长的（和报告 6.1 的结论一致）
  · 基座模型做不到稳定输出同一个 schema → 有明确的提升空间
  · 对错可以【程序自动判定】（JSON 能不能解析、每个字段对不对）→ 不再需要人工打分
  · 数据可以【程序化构造】→ 不需要手写 500 条

== 关键设计：训练集和评测集的难度分布故意不同 ==
  train: easy 50% / medium 40% / hard 10%
  eval : easy 20% / medium 40% / hard 40%
如果两边难度一样，模型可能只是"背下了模板"，测不出真本事。
难度错开之后，评测才能回答一个问题：它学到的是【抽取】还是【记忆】？

== 本文件里你要【看懂】的只有 1 处 ==
难度三档的定义（easy / medium / hard 各自怎么造、为什么要这么分）。
其余（随机池、模板、写文件）都是脚手架。

== 产物 ==
  data_domain/train.jsonl   约 1080 条
  data_domain/eval.jsonl    约 120 条
  data_domain/stats.json    字段分布统计（用于检查多样性）
"""
import os, json, random

OUT_DIR = r"C:\code\p0-python\data_domain"   # 输出目录
N_TOTAL = 1200                                # 总共造多少条
SEED = 42                                     # 固定随机种子 → 可复现

# ══════════════════════ 【直接复制】取值池 ══════════════════════
# 这些池子就是"世界"，程序从里面随机抽值，拼出一条 JD
COMPANIES = ["字节跳动", "腾讯", "阿里巴巴", "美团", "百度", "京东", "网易", "快手",
             "小米", "华为", "商汤科技", "旷视科技", "智谱AI", "月之暗面", "MiniMax",
             "百川智能", "科大讯飞", "地平线", "蔚来汽车", "小红书"]
CITIES = ["北京", "上海", "深圳", "杭州", "广州", "成都", "南京", "武汉", "西安", "苏州"]
TITLES = ["大模型算法工程师", "NLP算法工程师", "机器学习工程师", "AI应用开发工程师",
          "算法实习生", "数据挖掘工程师", "推荐算法工程师", "CV算法工程师",
          "大模型应用开发工程师", "多模态算法工程师"]
SKILLS = ["Python", "PyTorch", "TensorFlow", "大模型微调", "LoRA", "RAG", "Agent",
          "Transformer", "NLP", "计算机视觉", "推荐系统", "知识图谱", "C++", "SQL",
          "LangChain", "vLLM", "模型量化", "Prompt工程"]

# 同一个意思的多种说法 —— 用来让文本不那么像模板
EDU_PHRASES = {
    "本科": ["本科及以上学历", "统招本科以上", "本科以上学历"],
    "硕士": ["硕士及以上学历", "硕士研究生以上", "硕士以上学历"],
    "博士": ["博士学历", "博士研究生"],
    "不限": ["学历不限", "不限学历"],
}
CITY_PHRASES = ["工作地点：{c}", "base在{c}", "工作地为{c}", "办公地点：{c}"]

# 字段顺序写死 —— 训练目标里的 JSON 键顺序必须每次都一样
FIELDS = ["company", "title", "city", "education", "experience_years",
          "skills", "salary_min_k", "salary_max_k"]

# 给模型的固定指令（每次一字不差 —— 这正是让它学会这个 schema 的关键）
INSTRUCTION = ("从下面的招聘信息中抽取字段并输出 JSON，字段：company, title, city, "
               "education, experience_years, skills(数组), salary_min_k, salary_max_k。"
               "未提到的字段填 null。")


def make_record(rng, difficulty):
    """造一条样本：返回 (JD 文本, 期望的 JSON 字典, 难度标签)。"""
    # ① 先随机决定这条有哪些字段 —— 这就是难度的来源
    rec = {}
    rec["company"] = rng.choice(COMPANIES)
    rec["title"] = rng.choice(TITLES)
    rec["city"] = rng.choice(CITIES)
    rec["education"] = rng.choice(["本科", "硕士", "博士", "不限"])
    # 经验年限：easy/medium 会明写，hard 有时不写
    exp = rng.choice([0, 1, 2, 3, 5, 8])
    # 技能：2~5 个，保持顺序稳定
    k = rng.randint(2, 5)
    sk = rng.sample(SKILLS, k)
    # 薪资：月薪（千元），保证下限 < 上限
    smin = rng.choice([10, 15, 20, 25, 30, 35, 40, 50])
    smax = smin + rng.choice([10, 15, 20, 25, 30])

    rec["experience_years"] = exp
    rec["skills"] = sk
    rec["salary_min_k"] = smin
    rec["salary_max_k"] = smax

    # ② 拼 JD 文本 —— 按难度用不同句式
    parts = []

    if difficulty == "easy":
        # 最简单：字段挨个明写，语序标准
        parts.append("%s招聘%s。" % (rec["company"], rec["title"]))
        parts.append(rng.choice(CITY_PHRASES).format(c=rec["city"]) + "。")
        parts.append("要求%s，%s年工作经验。" % (rng.choice(EDU_PHRASES[rec["education"]]), exp))
        parts.append("熟悉%s。" % "、".join(sk))
        parts.append("月薪%d-%dK。" % (smin, smax))

    elif difficulty == "medium":
        # 中等：句式打乱、顺序变化、名词换成同义说法
        order = rng.sample(["city", "edu", "exp", "skill", "salary"], 5)  # 随机排列
        parts.append("%s因业务发展，现招聘%s一名。" % (rec["company"], rec["title"]))
        for item in order:
            if item == "city":
                parts.append(rng.choice(CITY_PHRASES).format(c=rec["city"]) + "。")
            elif item == "edu":
                parts.append("学历要求：%s。" % rng.choice(EDU_PHRASES[rec["education"]]))
            elif item == "exp":
                parts.append("需有%d年相关经验。" % exp)
            elif item == "skill":
                parts.append("技术栈涉及%s等。" % "、".join(sk))
            else:
                parts.append("薪资范围%d-%dK/月。" % (smin, smax))

    else:  # hard
        # 难：起承转合更绕；并且有一半概率故意【不写某些字段】→ 期望输出 null
        parts.append("%s的%s团队正在扩招。" % (rec["company"], rec["title"]))
        parts.append("该岗位" + rng.choice(CITY_PHRASES).format(c=rec["city"]) + "，")
        # 故意丢掉一部分字段：每条随机丢 1~2 个
        drop = set(rng.sample(["edu", "exp", "skill", "salary"], rng.randint(1, 2)))
        tail = []
        if "edu" not in drop:
            tail.append("希望候选人为%s" % rng.choice(EDU_PHRASES[rec["education"]]))
        if "exp" not in drop:
            tail.append("具备%d年以上经验" % exp)
        if "skill" not in drop:
            tail.append("熟练掌握%s" % "、".join(sk))
        parts.append("，".join(tail) + "。")
        if "salary" not in drop:
            parts.append("待遇%d-%dK。" % (smin, smax))
        else:
            # 薪资没写的时候，补一句无关的话，让文本更像真实 JD（也增加噪音）
            parts.append(rng.choice(["欢迎投递简历。", "团队氛围好。", "该岗位有转正机会。"]))

        # 被丢掉的字段，期望值改成 None（训练它输出 null）
        if "edu" in drop:
            rec["education"] = None
        if "exp" in drop:
            rec["experience_years"] = None
        if "skill" in drop:
            rec["skills"] = None
        if "salary" in drop:
            rec["salary_min_k"] = None
            rec["salary_max_k"] = None

    jd = "".join(parts)
    # 按下固定顺序重排字典 —— 保证每条 JSON 的键顺序一致
    ordered = {f: rec[f] for f in FIELDS}
    return jd, ordered, difficulty


def to_messages(jd, gold):
    """把一条样本转成 chat 格式（和 d28 的格式完全一致）。"""
    return {"messages": [
        {"role": "user", "content": INSTRUCTION + "\n\n招聘信息：" + jd},
        # 期望回答 = JSON 字符串。ensure_ascii=False 保证中文不被转义成 \uXXXX
        {"role": "assistant", "content": json.dumps(gold, ensure_ascii=False)},
    ]}


def build(n, easy_ratio, medium_ratio, seed):
    """按给定比例造 n 条，返回消息列表 + 难度标签列表。

    ⚠️ seed 必须不同！如果训练集和评测集用同一个种子，
       两边会生成出一模一样的前 n 条 → 评测集就成了训练集的子集 → 【数据泄漏】。
       那样测出来的分数是假的（模型见过答案）。
    """
    rng = random.Random(seed)          # 固定种子 → 每次跑出来的数据一模一样
    msgs, diffs = [], []
    seen = set()                       # 用来去重
    tries = 0                          # 防死循环
    while len(msgs) < n and tries < n * 50:
        tries += 1
        r = rng.random()               # 掷一个 0~1 的随机数决定难度
        if r < easy_ratio:
            d = "easy"
        elif r < easy_ratio + medium_ratio:
            d = "medium"
        else:
            d = "hard"
        jd, gold, d = make_record(rng, d)
        if jd in seen:                 # 文本完全一样 → 丢弃
            continue
        seen.add(jd)
        msgs.append(to_messages(jd, gold))
        diffs.append(d)
    return msgs, diffs


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── ① 训练集：easy 50% / medium 40% / hard 10%（种子 SEED）──
    n_train = int(N_TOTAL * 0.9)
    train_msgs, train_diffs = build(n_train, 0.50, 0.40, SEED)

    # ── ② 评测集：easy 20% / medium 40% / hard 40% ★ 难度分布故意不同、种子也不同 ──
    eval_msgs, eval_diffs = build(N_TOTAL - n_train, 0.20, 0.40, SEED + 1)

    # ── ②b 查重：确认评测集里没有任何一条出现在训练集里（防数据泄漏）──
    train_keys = set(json.dumps(m, ensure_ascii=False, sort_keys=True) for m in train_msgs)
    overlap = sum(1 for m in eval_msgs
                  if json.dumps(m, ensure_ascii=False, sort_keys=True) in train_keys)
    print("训练集与评测集重叠条数 = %d  %s" % (overlap, "✅ 无泄漏" if overlap == 0 else "❌ 有泄漏！"))

    # ── ③ 写文件 ──
    for name, data in [("train", train_msgs), ("eval", eval_msgs)]:
        path = os.path.join(OUT_DIR, "%s.jsonl" % name)
        with open(path, "w", encoding="utf-8") as f:
            for m in data:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        print("%-6s %5d 条 → %s" % (name, len(data), path))

    # ── ④ 多样性统计（用来自查"数据是不是太单一"）──
    from collections import Counter
    stats = {}
    for name, data, diffs in [("train", train_msgs, train_diffs),
                              ("eval", eval_msgs, eval_diffs)]:
        golds = [json.loads(m["messages"][1]["content"]) for m in data]
        c = lambda f: len(set(json.dumps(g[f], ensure_ascii=False) for g in golds))
        stats[name] = {
            "n": len(data),
            "difficulty": dict(Counter(diffs)),
            "distinct_company": c("company"), "distinct_title": c("title"),
            "distinct_city": c("city"),
            "distinct_education": c("education"),
            "distinct_exp": c("experience_years"),
            "distinct_skill_combo": c("skills"),
            "null_rate_skills": round(sum(g["skills"] is None for g in golds) / len(golds), 3),
        }
    with open(os.path.join(OUT_DIR, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\n=== 多样性统计 ===")
    for name in ["train", "eval"]:
        s = stats[name]
        print("[%s] %d 条 | 难度 %s" % (name, s["n"], s["difficulty"]))
        print("     公司 %d 种 / 岗位 %d 种 / 城市 %d 种 / 学历 %d 种 / 经验 %d 种 / 技能组合 %d 种"
              % (s["distinct_company"], s["distinct_title"], s["distinct_city"],
                 s["distinct_education"], s["distinct_exp"], s["distinct_skill_combo"]))
        print("     技能为 null 的比例 = %.1f%%" % (s["null_rate_skills"] * 100))

    # ── ⑤ 打印一条样例，让你肉眼确认数据长什么样 ──
    print("\n=== 训练集第 1 条 ===")
    print(msgs_demo(train_msgs[0]))


def msgs_demo(m):
    """把一条样本打印成好读的形式。"""
    lines = []
    for x in m["messages"]:
        lines.append("【%s】%s" % (x["role"], x["content"]))
    return "\n".join(lines)


if __name__ == "__main__":
    main()
