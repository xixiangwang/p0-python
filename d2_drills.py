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
sum = 0
for i in range(101):
    sum += i
print(f"求和={sum}")

# while版本
sum = 0
i = 1
while i<=100:
    sum += i
    i += 1
print(f"求和={sum}")

# 打印三角形
n = int(input("请输入大小："))
for i in range(1,n+1):
    print("*" * i)     #竟然可以这样