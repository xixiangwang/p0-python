# 2.去重但保持原顺序 ← 这题是重点
# 输入 [3,1,3,2,1,4] → 期望 [3,1,2,4]

def del_same_keep_order(nums):
    d = {}
    for n in nums:
        d[n] = d.get(n,0) + 1

    new_nums = []
    for n in nums:
        if d[n] > 0:
            new_nums.append(n)
            d[n] = 0
    return new_nums

print(del_same_keep_order([3,1,3,2,1,4]))
print(del_same_keep_order([]))


# 3.两个 list 求交集
# a=[1,2,3,4]、b=[3,4,5,6] → 期望 [3,4]
# 写两遍：set(a) & set(b) 版 + 手写循环版（遍历 a，if x in b 就收）

# set版本
def jiaoji_set(a,b):
    a1,b1 = set(a),set(b)             # 只学了set才能用交&，并|，差-,把list转成set
    return list(a1 & b1)
print(jiaoji_set([1,2,3,4],[3,4,5,6]))

# 循环版本
def jiaoji_loop(a,b):
    nums = []
    for n in a:
        for m in b:
            if n == m:
                nums.append(n)
    return nums
print(jiaoji_loop([1,2,3,4],[3,4,5,6]))



# 4.list 去重的两种写法（set vs 手写循环）
# 输入 [3,1,3,2,1,4] → set 版 [1,2,3,4]、循环版 [3,1,2,4]

# set版
def del_same_set(nums):
    return list(set(nums))
print(del_same_set([3,1,3,2,1,4]))

# 循环版
def del_same_loop(nums):
    d = {}
    for n in nums:
        d[n] = d.get(n,0) + 1

    new_nums = []
    for n in nums:
        if d[n] > 0:
            new_nums.append(n)
            d[n] = 0
    return new_nums
print(del_same_loop([3,1,3,2,1,4]))



# 5.用 dict 统计一段话里每个单词出现次数（按空格切）
# 输入 "the cat the dog the bird" → 期望 {'the':3, 'cat':1, 'dog':1, 'bird':1}
def counts(s):
    d = {}
    for n in s.split():
        d[n] = d.get(n,0) + 1
    return d

print(counts("the cat the dog the bird"))
print(counts("a   b  c"))

