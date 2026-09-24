# ===========================2.1 多个返回值 ====================================
# C 只能返回一个，Python 自动打包成元组
def min_max(nums):
    return min(nums),max(nums)
print(min_max([3,1,4,1,5]))         # (1, 5)

# ★ 更常用：直接拆开接
lo,hi = min_max([3,1,4,1,5])
print(lo,hi)                        # 1 5


# ========================2.2 默认参数===============================
# 规则：有默认值的参数必须放后面，def f(a=1, b): 直接语法错误。
def greet(name,greeting="你好"):
    return greeting + "，" + name
 
print(greet("wxx"))                #  你好, wxx
print(greet("wxx","早上好"))       # 早上好, wxx



# ========================2.3 关键字参数==========================
# 就是你写 key=... 的那个语法，现在知道它是什么了
# sorted(x, key=abs) 里的 key=、reverse=True 都是这个 —— sorted 定义了 key 和 reverse 两个可选参数。
def area(width,height):
    return width * height

print(area(3,4))                       # 12  按位置传
print(area(height=4,width=3))          # 12  按名字传，顺序反过来也对



# =============================2.4 局部与全局 ========================
# C 里函数里赋值的变量默认是全局的，Python 反过来 —— 函数里赋值就是新建局部变量。想改外面的得写 global count（先知道有这回事，别用）。
# 简单说就是，函数里的变量是局部变量
count = 0               # 全局
def f():
    count = 10          # ★ 这是新建的局部变量，跟外面那个没关系！
    print(count)

f()                     # 10
print(count)            # 0    ← 外面没变



# ============================2.5 列表推导式=====================================
#  4 行压成 1 行
# ⭐读法从左往右：[ 要什么 for 从哪拿 if 什么条件才要 ]

# 你已经会的写法
res = []
for n in [1,2,3,4,5,6]:
    if n % 2 == 0:
        res.append(n)
print(res)                                       

# 推导式
res = [n for n in [1,2,3,4,5,6] if n % 2 == 0]
print(res)                                           # 结果一样 [2,4,6]

print([n*n for n in range(1,6)])                     # [1, 4, 9, 16, 25]
print([s.upper() for s in ["a","b"]])                # ['A', 'B']  ,s.upper()应该是大写



# =============================2.6 字典推导式====================
# 唯一区别是 {} 里写 键: 值
print({n:n*n for n in range(1,4)})             # {1: 1, 2: 4, 3: 9}

nums = ["a","b","c"] 
print({n:i for i,n in enumerate(nums)})        # {'a': 0, 'b': 1, 'c': 2}



# =============================2.7 lambda=========================
# ⭐读法 lambda 参数: 返回值，只能写一个表达式
# def 能写多条语句、能写文档字符串、能起名字反复用；lambda 只有一行、用完就扔。能用 def 的地方优先 def，lambda 只适合"就在这儿用一次"的小规则。
# key= 不是通用参数！ 它只属于 sorted() / max() / min() 这三个"需要知道怎么比大小"的函数 —— 因为只有它们内部要拿这个规则去比较。

def by_second(p):
    return p[1]
print(by_second([1,2,3]))
# 等价于
by_second = lambda p:p[1]
print(by_second([1,2,3]))



# ==============================2.8 key= 换比较规则======================

# ⚠️ key=abs 里 abs 后面没有括号 —— 传的是"函数本身"，不是"调用结果"。
nums = [-1,3,-2]
print(sorted(nums))                # [-2, -1, 3]   默认拿元素本身比
print(sorted(nums,key = abs))      # [-1, -2, 3]   先求绝对值再比（1 < 2 < 3）

# 你为了"按次数排"把数据改造成 [次数, 字符]；用 key= 就不用改数据，只改比较规则。
pairs = [[5, 'b'], [4, 'a'], [3, 'c']]
print(sorted(pairs,key = lambda p:p[1]))      # 按字符排
print(sorted(pairs,key = lambda p:p[0]))      # 按次数排



# ================================2.9 拆开那一行================================
# 就是之前d5_wordcount里不懂的那个
# 将d.items通过值排序并反转从大到小排，最后取前三个
d = {"a":3,"b":2,"c":6}
print(sorted(d.items() ,  key=lambda kv: kv[1] ,  reverse=True )[:3])



# ==============================2.10 动手（就这两件）======================
# 验收：合上，把 wordcount_short 默写出来，跑出 [('b',5), ('a',4), ('c',3)]
def wordcount_short(s):
    d = {}
    for n in s:
        d[n] = d.get(n,0) + 1

    return sorted(d.items(),key=lambda p:p[1],reverse=True)[:3]

print(wordcount_short("abcdaaadcbbbbc"))
print(wordcount_short("aabb"))
print(wordcount_short("aa"))
print(wordcount_short(""))