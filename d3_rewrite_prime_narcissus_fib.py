# ===== d3_rewrite_prime_narcissus_fib.py =====
# 本文件包含：① 重写素数判断（标志位）② 重写水仙花数 ③ 重写斐波那契前 20 项
# 相关：第一版（还含 100 内质数表）→ d2_loops_table_prime.py · d3_factorial_narcissus_fib.py
# ===============================================

# 重写素数  （标志位）
n = int(input("请输入一个数字："))
is_prime = True
for i in range(2,n):
    if n % i == 0:
        is_prime = False
        break
if is_prime:
    print("素数")
else:
    print("不是素数")

# 重写水仙花数
n = int(input("请输入一个三位数："))
bai = n // 100
shi = n % 100 //10
ge = n % 10
total = bai*bai*bai + shi*shi*shi + ge*ge*ge
if total == n:
    print("是水仙花数")
else:
    print("不是水仙花数")

# 重写斐波那契前20项
a,b = 0,1
for i in range(20):
    print(a,end=" ")
    a,b = b,a+b
print()