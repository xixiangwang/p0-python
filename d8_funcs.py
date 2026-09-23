# ===========================判断质数、求最大公约数，必须写成带参数的函数==========================

# 判断质数
def is_prime(n):
    if n < 2:
        return False
    for i in range(2,n):
        if n % i == 0:
            return False
    return True

print([is_prime(x) for x in [2, 1, 9, 17, 0, 4, 13]])



# 求最大公约数
def gcd(a,b):
    max_gcd = 1
    for i in range(2,max(a,b)+1):
        if a % i == 0 and b % i == 0:
            max_gcd = i
    return max_gcd

print([gcd(a,b) for a,b in [(12,18),(7,13),(100,75),(5,0)]])

        