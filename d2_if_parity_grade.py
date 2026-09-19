# ===== d2_if_parity_grade.py =====
# 本文件包含：① 奇偶判断（n % 2 == 0）② 分数等级（if / elif / else 多分支链）
# 相关：冒号 + 缩进是真语法；C 的 if 链对照 →《C转Python迁移手册》
# ===================================

# 练习 1：判断奇偶
n = int(input("请输入一个整数："))
if n % 2 == 0:
    print("偶数")
else:
    print("奇数")

# 练习 2：分数等级
score = float(input("请输入你的成绩："))
if score >= 90:
    print("优秀")
elif score >= 80:
    print("良好")
elif score >= 60:
    print("及格")
else:
    print("不及格")