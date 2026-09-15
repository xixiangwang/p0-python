# n 的阶乘
n = int(input("输入一个数："))
total = 1
for i in range(1,n+1):
    total *= i
print(f"{n}的阶乘={total}")

# 水仙花数（教两个新符号）
for i in range(100,1000):
    bai = i // 100
    shi = i % 100 // 10
    ge  = i % 10

    total = bai * bai * bai + shi * shi * shi + ge * ge * ge  
    if total == i:
        print(f"{i}")

# 斐波那契前 20 项,我记得是递归
a,b = 0,1
for i in range(20):
    print(a,end=" ")
    a,b = b,a+b
print()
# 100 以内质数表
for i in range(2,101):
    zhishu = True
    for j in range(2,i):
        if i % j == 0:
            zhishu = False
            break
    if zhishu:
        print(i,end = " ")
print()
