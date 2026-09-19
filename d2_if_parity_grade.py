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