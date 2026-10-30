# -*- coding: utf-8 -*-
# 最小 ReAct 循环：思考 → 行动 → 观察，带两个终止条件
#
# 用法：
#   python agent_react.py                      进交互模式，连续提问
#   python agent_react.py "你的问题"            一次性提问
#   输入 q 或直接回车退出

import sys


def fake_search(q):
    """假工具：假装是一个知识库检索（真实版本这里换成向量库或网页搜索）"""
    if "天气" in q:
        return "没找到相关内容。"
    if "固定" in q:
        return "固定长度切分不看语义，会在句子中间切断，块首尾是半句。"
    if "语义" in q:
        return "语义切分按段落和句子边界切，块内语义完整。"
    if "向量" in q:
        return "向量检索用余弦相似度，暴力扫描是 O(N·d)。"
    return "没找到相关内容。"


TOOLS = {"search": fake_search}


def decide(question, history, step):
    """
    每一轮的第一步：要不要动、动哪个工具、传什么参数。
    真实版本的 Agent，这一步是把 question + history 喂给 LLM，让它输出下一步动作。
    """
    if not history:
        return "search", question
    if "没找到" not in history[-1]["obs"]:
        return None, None
    if step == 1:
        return "search", question
    return "search", question + " 语义"


def finish(question, history):
    """收尾：把收集到的观察拼成答案；一次没搜到就老实说不知道"""
    good = [h["obs"] for h in history if "没找到" not in h["obs"]]
    if not good:
        return "查了 %d 次都没找到资料，只能说不知道。" % len(history)
    return "查了 %d 次，回答：%s" % (len(history), " ".join(good))


def react(question, tools, max_steps=3, verbose=True):
    """最小 ReAct 循环"""
    history = []
    for step in range(1, max_steps + 1):
        action, arg = decide(question, history, step)
        if verbose:
            print("  [第 %d 轮] 决策 = %s(%s)" % (step, action, arg))

        if action is None:
            return finish(question, history)

        obs = tools[action](arg) if action in tools else "没有这个工具"
        if verbose:
            print("            观察 = %s" % obs)
        history.append({"step": step, "action": action, "arg": arg, "obs": obs})

    if verbose:
        print("  [轮次用完 -> 强制终止]")
    return finish(question, history) + "（已达最大轮次）"


def main():
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
        print("问题：" + q)
        print(react(q, TOOLS))
        return

    print("输入问题后回车执行；直接回车或输入 q 退出。")
    while True:
        try:
            q = input("\n问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            break
        if q in ("", "q", "quit", "exit"):
            print("已退出。")
            break
        print(react(q, TOOLS))


if __name__ == "__main__":
    main()
