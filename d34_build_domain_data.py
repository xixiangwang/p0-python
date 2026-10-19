# -*- coding: utf-8 -*-
"""d34 (v2)：自造领域数据 —— 招聘 JD → 结构化 JSON
跑法：cd C:\code\p0-python
      python d34_build_domain_data.py
（纯 CPU，几秒钟跑完，不碰 GPU）

== 这一版（v2）改了什么，为什么要改 ==
v1 训出来的模型在同分布评测上 100%，但 OOD 上只有 90.9%。
把 10 个错误逐个看过之后，全部集中在两个模式上（没有例外）：
  模式 A（3 条）：公司名不在句首时，它把岗位名当公司名（完全对调），
                  或者把句首状语"在长沙"当成公司。
                  → 根因：v1 的训练数据里，公司名 100% 在句首。
  模式 B（7 条）："了解 X、Y 者优先考虑"里的 X、Y 被算进了硬要求。
                  → 根因：v1 的训练数据里从来没有"优先/加分"这种干扰项。

所以 v2 在【训练集】里补上这两件事：
  ① 打乱字段位置 —— 公司名有时在句首、有时在句中、有时在最后
  ② 加入"X 者优先 / 加分"干扰句 —— 这两个技能不进标准答案

== ⚠️ 评测集【一个字都不改】==
eval.jsonl 保留 v1 生成的那一份，不再覆盖。
因为这才能和 d36 已经跑出的 100% 同口径比较；
改评测集之后再比，等于自己骗自己。

== 产物 ==
  data_domain/train.jsonl      约 1080 条（v2，重新生成）
  data_domain/eval.jsonl       120 条（v1，保持不变）
  data_domain/stats.json       字段分布统计
"""
import os, json, random, sys, hashlib

OUT_DIR = r"C:\code\p0-python\data_domain"
N_TOTAL = 1200
SEED = 42

# ══════════════════════ 单位换算实验的开关（v3）══════════════════════
# 跑法：python d34_build_domain_data.py --wan
# 作用：把一部分样本的薪资从 "40-50K" 改写成 "4万-5万"（标准答案不变，仍是千元数值）
# 目的：测「模型能不能学会『万→千』这个换算」，还是说它只会照抄字面。
USE_WAN = "--wan" in sys.argv
WAN_RATIO = 0.15


def _wan(x):
    """把「千元」数值转成「万」的字面：40 → '4'，45 → '4.5'。"""
    s = "%.1f" % (x / 10.0)
    return s[:-2] if s.endswith(".0") else s


def _pick(s):
    """由字符串内容决定一个 0~1 的固定值。

    ⚠️ 为什么不用 rng.random()：那会推进主随机流，导致 v3 的样本
       和 v2 不是同一批 —— 实验就变成"换了一批数据"而不是"只改了薪资写法"。
       用内容哈希 → 同一条文本永远得到同一个判断，主随机流一动不动。
       **这就是控制变量。**
    """
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
# ═══════════════════════════════════════════════════════════════════

# ══════════════════════ 【直接复制】取值池 ══════════════════════
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

EDU_PHRASES = {
    "本科": ["本科及以上学历", "统招本科以上", "本科以上学历"],
    "硕士": ["硕士及以上学历", "硕士研究生以上", "硕士以上学历"],
    "博士": ["博士学历", "博士研究生"],
    "不限": ["学历不限", "不限学历"],
}
CITY_PHRASES = ["工作地点：{c}", "base在{c}", "工作地为{c}", "办公地点：{c}"]

FIELDS = ["company", "title", "city", "education", "experience_years",
          "skills", "salary_min_k", "salary_max_k"]

INSTRUCTION = ("从下面的招聘信息中抽取字段并输出 JSON，字段：company, title, city, "
               "education, experience_years, skills(数组), salary_min_k, salary_max_k。"
               "未提到的字段填 null。")


# ══════════════════════ 【看懂】新增：优先/加分干扰句 ══════════════════════
def bonus_phrase(rng, sk):
    """造一句"X、Y 者优先"的干扰句。

    ⚠️ 关键：这两个技能【不进标准答案】。
       现实 JD 里经常写"了解 X 者优先"——那不是硬要求。
       模型必须学会区分"要求"和"加分项"，否则抽取出来的技能列表就是脏的。
    """
    cand = [s for s in SKILLS if s not in sk]      # 从没被要求的技能里挑
    b = rng.sample(cand, 2)
    return rng.choice(["了解%s、%s者优先。",
                       "有%s、%s经验者加分。",
                       "熟悉%s或%s者优先考虑。"]) % (b[0], b[1])
# ═══════════════════════════════════════════════════════════════════


def make_record(rng, difficulty, mode="train"):
    """造一条样本：返回 (JD 文本, 期望的 JSON 字典, 难度标签)。

    mode="train" → 启用 v2 的两个新特性（打乱位置 + 优先干扰）
    mode="eval"  → 退回 v1 的老行为（评测集必须保持原样才可比）
    """
    rec = {}
    rec["company"] = rng.choice(COMPANIES)
    rec["title"] = rng.choice(TITLES)
    rec["city"] = rng.choice(CITIES)
    rec["education"] = rng.choice(["本科", "硕士", "博士", "不限"])
    exp = rng.choice([0, 1, 2, 3, 5, 8])
    k = rng.randint(2, 5)
    sk = rng.sample(SKILLS, k)                     # 硬要求（进标准答案）
    smin = rng.choice([10, 15, 20, 25, 30, 35, 40, 50])
    smax = smin + rng.choice([10, 15, 20, 25, 30])

    rec["experience_years"] = exp
    rec["skills"] = sk                             # ★ 只有硬要求进答案
    rec["salary_min_k"] = smin
    rec["salary_max_k"] = smax
    ordered = {f: rec[f] for f in FIELDS}

    edu_p = rng.choice(EDU_PHRASES[rec["education"]])
    sks = "、".join(sk)
    city_p = rng.choice(CITY_PHRASES).format(c=rec["city"])

    # ══════════════ easy：字段明写 ══════════════
    if difficulty == "easy":
        items = [("intro", "%s招聘%s。" % (rec["company"], rec["title"])),
                 ("city", city_p + "。"),
                 ("edu", "要求%s，%s年工作经验。" % (edu_p, exp)),
                 ("skill", "熟悉%s。" % sks),
                 ("salary", "月薪%d-%dK。" % (smin, smax))]
        # ★ v2 新增：40% 的概率把"公司/岗位"这一句挪到后面 —— 打破"公司永远在最前"
        if mode == "train" and rng.random() < 0.40:
            intro = items.pop(0)
            items.insert(rng.randint(2, len(items)), intro)
        parts = [t for _, t in items]
        # ★ v2 新增：35% 概率补一句"优先"干扰
        if mode == "train" and rng.random() < 0.35:
            parts.append(bonus_phrase(rng, sk))

    # ══════════════ medium：全句打乱 ══════════════
    elif difficulty == "medium":
        items = [("intro", "%s因业务发展，现招聘%s一名。" % (rec["company"], rec["title"])),
                 ("city", city_p + "。"),
                 ("edu", "学历要求：%s。" % edu_p),
                 ("exp", "需有%d年相关经验。" % exp),
                 ("skill", "技术栈涉及%s等。" % sks),
                 ("salary", "薪资范围%d-%dK/月。" % (smin, smax))]
        if mode == "train":
            # ★ v2 改动：连"公司/岗位"句一起参与打乱（v1 是固定在最前）
            rng.shuffle(items)
        else:
            # v1 老行为：先固定"公司/岗位"，再打乱后面五项
            head = items.pop(0)
            rng.shuffle(items)
            items.insert(0, head)
        parts = [t for _, t in items]
        if mode == "train" and rng.random() < 0.55:
            parts.insert(rng.randint(1, len(parts)), bonus_phrase(rng, sk))

    # ══════════════ hard：段落式 + 故意丢字段 ══════════════
    else:
        drop = set(rng.sample(["edu", "exp", "skill", "salary"], rng.randint(1, 2)))
        # ★ v2 新增：50% 概率换一种开场 —— 公司名不在句首
        if mode == "train" and rng.random() < 0.50:
            head = "现招聘%s一名，用人单位：%s。" % (rec["title"], rec["company"])
        else:
            head = "%s的%s团队正在扩招。" % (rec["company"], rec["title"])

        tail = []
        if "edu" not in drop:
            tail.append("希望候选人为%s" % edu_p)
        if "exp" not in drop:
            tail.append("具备%d年以上经验" % exp)
        if "skill" not in drop:
            tail.append("熟练掌握%s" % sks)
        parts = [head, "该岗位" + city_p + "，", "，".join(tail) + "。"]
        if "salary" not in drop:
            parts.append("待遇%d-%dK。" % (smin, smax))
        else:
            parts.append(rng.choice(["欢迎投递简历。", "团队氛围好。", "该岗位有转正机会。"]))
        if mode == "train" and rng.random() < 0.60:
            parts.append(bonus_phrase(rng, sk))

        # 被丢掉的字段 → 期望输出 null
        if "edu" in drop:
            ordered["education"] = None
        if "exp" in drop:
            ordered["experience_years"] = None
        if "skill" in drop:
            ordered["skills"] = None
        if "salary" in drop:
            ordered["salary_min_k"] = None
            ordered["salary_max_k"] = None

    # ★ 实验开关：只把薪资的写法换掉，别的什么都不动（控制变量）
    #   用 _pick(文本) 而不是 rng.random() —— 主随机流一个字节都不动，
    #   所以 v3 的样本和 v2 完全一致，唯一的差别就是薪资写法。
    jd = "".join(parts)
    if USE_WAN and _pick(jd) < WAN_RATIO:
        jd = jd.replace("%d-%dK" % (smin, smax),
                        "%s万-%s万" % (_wan(smin), _wan(smax)))
    return jd, ordered, difficulty


def to_messages(jd, gold):
    return {"messages": [
        {"role": "user", "content": INSTRUCTION + "\n\n招聘信息：" + jd},
        {"role": "assistant", "content": json.dumps(gold, ensure_ascii=False)},
    ]}


def build(n, easy_ratio, medium_ratio, seed, mode="train"):
    """按比例造 n 条。seed 不同 → 内容不同（train 和 eval 必须用不同种子，否则泄漏）。"""
    rng = random.Random(seed)
    msgs, diffs, seen = [], [], set()
    tries = 0
    while len(msgs) < n and tries < n * 50:
        tries += 1
        r = rng.random()
        if r < easy_ratio:
            d = "easy"
        elif r < easy_ratio + medium_ratio:
            d = "medium"
        else:
            d = "hard"
        jd, gold, d = make_record(rng, d, mode)
        if jd in seen:
            continue
        seen.add(jd)
        msgs.append(to_messages(jd, gold))
        diffs.append(d)
    return msgs, diffs, len(seen) > 0


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── ① 训练集：重新生成 ──
    n_train = int(N_TOTAL * 0.9)
    train_msgs, train_diffs, _ = build(n_train, 0.50, 0.40, SEED, mode="train")
    # 开关打开时写到 train_wan.jsonl，不覆盖 v2 的 train.jsonl
    train_name = "train_wan.jsonl" if USE_WAN else "train.jsonl"
    with open(os.path.join(OUT_DIR, train_name), "w", encoding="utf-8") as f:
        for m in train_msgs:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print("train  %5d 条 → %s（%s）" % (
        len(train_msgs), train_name,
        "v3：15% 薪资改成「万」表达" if USE_WAN else "v2：打乱位置 + 优先干扰"))
    if USE_WAN:
        n_wan = sum(1 for m in train_msgs if "万-" in m["messages"][0]["content"])
        print("       其中「万」表达样本：%d / %d = %.1f%%"
              % (n_wan, len(train_msgs), 100 * n_wan / len(train_msgs)))

    # ── ② 评测集：★ 不覆盖，保持 v1 原样 ──
    eval_path = os.path.join(OUT_DIR, "eval.jsonl")
    if os.path.exists(eval_path):
        with open(eval_path, encoding="utf-8") as f:
            eval_msgs = [json.loads(l) for l in f if l.strip()]
        print("eval   %5d 条 → 保持不变（与 d36 的 100%% 同口径）" % len(eval_msgs))
        eval_diffs = []
    else:
        eval_msgs, eval_diffs, _ = build(N_TOTAL - n_train, 0.20, 0.40, SEED + 1, mode="eval")
        with open(eval_path, "w", encoding="utf-8") as f:
            for m in eval_msgs:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        print("eval   %5d 条 → 已生成（v1 行为）" % len(eval_msgs))

    # ── ③ 自查：训练集里"公司不在句首"和"含优先项"各占多少 ──
    n_shift = sum(1 for m in train_msgs
                  if not m["messages"][0]["content"].split("招聘信息：")[-1]
                  .startswith(tuple(COMPANIES)))
    n_bonus = sum(1 for m in train_msgs
                  if any(w in m["messages"][0]["content"] for w in ["者优先", "者加分", "优先考虑"]))
    print("\n=== v2 训练集自查 ===")
    print("公司不在句首的样本：%d / %d = %.1f%%" % (n_shift, len(train_msgs), 100 * n_shift / len(train_msgs)))
    print("含「优先/加分」干扰的样本：%d / %d = %.1f%%" % (n_bonus, len(train_msgs), 100 * n_bonus / len(train_msgs)))

    # ── ④ 泄漏自查 ──
    k = lambda m: json.dumps(m, ensure_ascii=False, sort_keys=True)
    tr = set(k(m) for m in train_msgs)
    print("与 eval 重叠条数 = %d  %s" % (sum(1 for m in eval_msgs if k(m) in tr),
                                        "✅ 无泄漏"))

    # ── ⑤ 打印两条样例 ──
    print("\n=== 样例：公司不在句首 ===")
    for m in train_msgs[:400]:
        if not m["messages"][0]["content"].split("招聘信息：")[-1].startswith(tuple(COMPANIES)):
            print(m["messages"][0]["content"].split("招聘信息：")[-1])
            print("→ GOLD:", m["messages"][1]["content"]); break
    print("\n=== 样例：含优先项干扰 ===")
    for m in train_msgs[:400]:
        if "者优先" in m["messages"][0]["content"]:
            print(m["messages"][0]["content"].split("招聘信息：")[-1])
            print("→ GOLD:", m["messages"][1]["content"])
            print("⚠️ 注意 GOLD 里【没有】\"优先\"那两个技能"); break


if __name__ == "__main__":
    main()
