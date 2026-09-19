# ===== d2_loops_table_prime.py =====
# 本文件包含：① 九九乘法表（嵌套 for + end="\t"）② 质数判断（标志位）③ 1~100 求和（for 版 + while 版各一遍）④ 打印三角形（字符串乘法）
# 相关：标志位的同题重写 → d3_rewrite_prime_narcissus_fib.py
# =====================================

# 九九乘法表（双重循环）
for i in range(1,10):
    for j in range(1,i+1):
        print(f"{j}*{i}={i*j}",end = "\t") #end="\t" 的意思是"打印完不换行，用制表符接上
    print()   # 这个很重要

n = int(input("输入一个数："))
is_prime = True    # 最重要的是要有一个标志位
for i in range(2,n):
    if n % i == 0:
        is_prime = False
        break
if is_prime:    #Python 会去看 is_prime 这个变量装的是什么：

                #装的是 True → 走进 if 里面
                #装的是 False → 走进 else 里面
    print("质数")
else:
    print("不是质数")
 
# 1–100 求和
# for版本
total = 0
for i in range(1,101):
    total += i
print(f"求和={total}")

# while版本
total = 0
i = 1
while i<=100:
    total += i
    i += 1
print(f"求和={total}")

# 打印三角形
n = int(input("请输入大小："))
for i in range(1,n+1):
    print("*" * i)     # 竟然可以这样