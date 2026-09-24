# ======================版本一：字符串处理====================
# 1. f.read()：读数据
# 2. s.split("\n")：去\n
# 3. s[1:]: 去掉第一项
with open("scores.csv","r",encoding="utf-8") as f:
    lines = f.read().split("\n")[1:]                # s.split("\n"),字符串操作，去掉

data_lines = []
for i in range(len(lines)):                     # 这里应该先分元组,这样方便将score变成int
    parts = lines[i].split(",")
    data_lines.append((parts[0],int(parts[1])))
print(data_lines)                               # 只是为了看变化，不是必要

def ave(n):
    return sum(n) / len(n)

scores = [s for name,s in data_lines]
print("人数：",len(scores))
print("平均分：",ave(scores))


# 版本一优化版
with open("scores.csv","r",encoding="utf-8") as f:
    lines = f.read().split("\n")[1:]                # s.split("\n"),字符串操作，去掉

scores = [int(line.split(",")[1]) for line in lines]       # 直接缩短成这一行

print("人数：",len(scores))
print("平均分：",ave(scores))


# 版本二：用csv模块
import csv

with open("scores.csv","r",encoding="utf-8") as f:
    rows = list(csv.reader(f))
    print(rows)                  # 可以忽略，也只是看变化

data = [(r[0],int(r[1])) for r in rows[1:]]              # 这里和方法一改进版一样，很重要
print(data)

scores = [line[1] for line in data]
print(scores)

print("人数：",len(scores))
print("平均分：",ave(scores))

