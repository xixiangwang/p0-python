# ========================= d9_file_except.py ==========================
# Day 7 知识点：文件读写 + 异常 + 常用库
# 对照 C： fopen / fgets / fclose  →  open + with（不用手动关）
#          返回错误码 check        →  try / except（出事就抛，能接的地方接）
# print(repr(...)) 里的 repr 是用来把 \n 显示出来的 —— 不然你只看到换行，不知道行尾带了 \n。

# =========================== 7.1 读文件 ===============================

# 先造一个文件出来，方便下面读（这本身就是"写文件"）
with open("demo.txt","w",encoding="utf_8") as f:
    f.write("第一行\n")
    f.write("第二行\n")
    f.write("第三行\n")


# 下面是三种读法:
# ⚠️ ②③ 读出来的每行"行尾都带 \n"，想干净要 line.strip()


# ① read() —— 整个文件读成一个字符串
with open("demo.txt","r",encoding="utf-8") as f:
    print(repr(f.read()))                          # → '第一行\n第二行\n第三行\n'

# ② readlines() —— 每行一个元素，组成 list
with open("demo.txt","r",encoding="utf-8") as f:
    print(f.readlines())                           # → ['第一行\n', '第二行\n', '第三行\n']

# ③ 直接 for —— 文件很大时最省内存，用这个
with open("demo.txt","r",encoding="utf-8") as f:
    for line in f:
        print(repr(line))                          # → '第一行\n' / '第二行\n' / '第三行\n'

# ④ with 会自动关文件 —— 实证
with open("demo.txt",encoding="utf-8"):
    pass
print(f.closed)                                    # → True    ← 出块就自动关了

f2 = open("demo.txt",encoding="utf-8")
print(f2.closed)                                   # → False   ← 不用 with 就得自己 close
f2.close()
print(f2.closed)                                   # → True


# =========================== 7.2 写文件 ===============================
# ⚠️ f.write() 不像 print 那样自动换行，\n 要自己写

# w 模式：先清空原文件，再写
with open("out.txt","w",encoding="utf-8") as f:
    f.write("AAAA\n")
    f.write("CCCC\n")
with open("out.txt","r",encoding="utf-8") as f:
    print(repr(f.read()))                           # 'AAAA\nCCCC\n'
with open("out.txt","w",encoding="utf-8") as f:
    f.write("BBBB\n")
with open("out.txt","r",encoding="utf-8") as f:
    print(repr(f.read()))                           # → 'BBBB\n'   ← 两个 A 没了，w 会清空

# a 模式：追加，不清空
with open("out.txt","a",encoding="utf-8") as f:
    f.write("CCCC\n")
with open("out.txt","r",encoding="utf-8") as f:
    print(repr(f.read()))                           # 'BBBB\nCCCC\n'




# ============================ 7.3 异常 ================================

# try = "这段可能会炸"      except X = "炸出 X 这种错时，我这样处理"
# 流程如下：
# try:
#     可能出事的事 
# except 某种错误:
#     出这种错时怎么办
# else:
#     没出错才跑这里
# finally:
#     不管怎样都跑这里

# try / except / else / finally 四段
def div(a,b):
    try:
        r = a/b
    except ZeroDivisionError as e:
        print("except 抓到：",e)
    else:
        print("else 没出错，结果是：",r)
    finally:
        print("finally 一定跑")

div(10,2)                   # → else 没出错，结果是 5.0
                            # → finally 一定跑
div(10,0)                   # → except 抓到: division by zero
                            # → finally 一定跑

# 一次抓多种异常
for bad in ["abc",[1,2,3]]:
    try:
        int(bad)
    except (ValueError,TypeError) as e:          # → 抓到 ValueError -> invalid literal for int() with base 10: 'abc'
        print("抓到",type(e).__name__,"->",e)    # → 抓到 TypeError -> int() argument must be a string, ... not 'list'


# raise：自己抛异常
def check_age(n):
    if n < 0:
        raise ValueError("年龄不能是负数:" + str(n))
    return n
try:
    check_age(-5)
except ValueError as e:
    print("抓到:",e)



# =========================== 7.4 常用库 ===============================

# import 三种写法
import math
print(math.sqrt(16))            # → 4.0

import math as m
print(m.pi)                     # → 3.141592653589793

from math import sqrt,pi
print(sqrt(25),pi)              # → 5.0 3.141592653589793


# math 常用
print(math.floor(3.7))         # → 3    向下取整
print(math.ceil(3.7))          # → 4    向上取整
print(math.gcd(12,18))         # → 6    ⭐ 标准库自带最大公约数
print(math.isqrt(17))          # → 4    向下取整的平方根


# random 常用
import random
random.seed(42)                            # 固定种子，结果可复现,42就是个随便挑的整数
print(random.randint(1,6))                 # → 6
print(random.choice(["a","b","c"]))        # → a
s = [1,2,3,4,5]
random.shuffle(s)                          # 原地打乱
print(s)                                   # → [2, 4, 5, 3, 1]


# os 常用
import os
print(os.path.exists("demo.txt"))
print(os.path.exists("zzz.txt"))


# json 常用：字典 <-> JSON 字符串
import json
d = {"name":"张三","scores":[85,90]} 
text = json.dumps(d,ensure_ascii=False)     # ensure_ascii=False 中文才不乱码
print(text)                                 # → {"name": "张三", "scores": [85, 90]}
print(json.loads(text))                     # → {'name': '张三', 'scores': [85, 90]}


# 清掉演示文件，别让它们混进 git
os.remove("demo.txt")
os.remove("out.txt")
print("演示文件已清理")