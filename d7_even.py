# 返回List中的偶数
def even_for(xs):      # 用 for + if + append
    d = []
    for x in xs:
        if x % 2 == 0:
            d.append(x)
    return d

print(even_for([1,2,3,4,5,6]))    # 期望 [2, 4, 6]
print(even_for([1,3,5]))          # 期望 []
print(even_for([]))               # 期望 []   ← 空表必测
print(even_for([-2,-1,0]))        # 期望 [-2, 0]



def even_comp(xs):     # 用列表推导式，一行
    
    return [x for x in xs if x % 2 == 0]

print(even_comp([1,2,3,4,5,6]))    # 期望 [2, 4, 6]
print(even_comp([1,3,5]))          # 期望 []
print(even_comp([]))               # 期望 []   ← 空表必测
print(even_comp([-2,-1,0]))        # 期望 [-2, 0]