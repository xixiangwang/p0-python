# 求 list 的最大 / 最小 / 平均值

# 最大
nums = [3, 1, 4, 1, 5, 9, 2, 6]
max_val = nums[0]
for n in nums:
    if n>max_val:
        max_val = n
print(f"最大={max_val}")
print(max(nums))

# 最小
min_val = nums[0]
for n in nums:
    if n<min_val:
        min_val = n
print(f"最小={min_val}")
print(min(nums))

# 平均值
total = 0
count = 0
for n in nums:
    total += n
    count += 1
ave = total / count
print(f"平均={ave}")
print(sum(nums) / len(nums))



# 创建
a = []                    # 空
b = [1,2,3]               # 直接给值
c = list(range(5))        # [0,1,2,3,4]
d = [0] * 3               # [0,0,0]
e = [1,"abc",3.14,True]   # ⚠️ C 做不到：list 能装不同类型
print(a,b,c,d,e)



# 取 / 改 / 查
print(b[0],b[-1],b[-2],len(b))
b[0] = 99
print(99 in b)      # True  ← 查存在



# 加
a = [1,2]
a.append(3)         # [1,2,3]        尾部加一个 ← 最常用
a.insert(0,99)      # [99,1,2,3]     插到下标 0
a.extend([7,8])     # [99,1,2,3,7,8] 把另一个 list 接上来
a + [0]             # 也能拼（返回新 list，不改 a）
print(a,a+[0],a)



# 删（这组容易混，仔细看）
a.pop()       # 删除最后一个，并返回值
a.pop(0)      # 删除下标0那个。也会返回值
a.remove(2)   # 删除第一个值为2的（按值删，不是按下标！）
del a[1]      # 删除下标为1的，不返回值



# 排序 —— sort() 和 sorted() 的区别，务必搞清
a = [3,1,2]
a.sort()       # 原地排，a的值也会变,但注意它的返回值是 None，不是排好序的列表
b = sorted(a)  # 返回排序好的新List，a不变
a.reverse()    # 原地反转
a[::-1]        # a[::-1] 是切片的三参数写法：a[起始 : 结束 : 步长]，1 不是"减一个"，是"方向朝左"​
               # 返回新 list，但如果不打印，不赋值，就只算出结果，然后立刻丢掉



# 切片 —— 今天最重要的新东西,左开右闭
a = [0,1,2,3,4,5]
a[1:3]      # [1,2]
a[:2]       # [0,1]
a[2:]       # [2,3,4,5]
a[::2]      # [0,2,4]
a[::-1]     # [5,4,3,2,1,0]



# 遍历（三种，都要会）,其中end=" ",和最后的print()不必要，只是为了好看
for x in a:                  # 只要元素 ← 最常用
    print(x,end=" ")
print()

for i,x in enumerate(a):     # 同时要下标和元素 ← 强烈推荐
    print(i,x,end=" ")
print()

for i in range(len(a)):      # C 式写法，能用但不地道
    print(a[i],end=" ")
print()



# 坑1：赋值 ≠ 拷贝
a = [1,2,3]
b = a          # 这不是拷贝，是给同一块内存起了个新名字（起别名）
b.append(4)
print(a)       # [1, 2, 3, 4]  ← a 也变了！

# 真拷贝三种写法（结果一样，都跟 a 脱钩）
c=a[:]
c=a.copy()
c=list(a)

# b=a就是a和b都是这一个List，无论动谁，a和b都跟这变，而b=a.copy()才是创建了新的独立List
# copy() 是浅拷贝，只拷贝最外层，比如二维数组
# 但是需要注意，只有对a和b改内容才会相互影响，但换掉整个对象就不会影响，比如下面
a = [1, 2, 3]
b = a

b.append(4)      # 改内容（原地修改）→ a 也跟着变
print(a)         # [1, 2, 3, 4]     ✅ 和你说的一样

b = [9, 9]       # 换对象（重新绑定）→ a 不变
print(a)         # [1, 2, 3, 4]     ← 没跟着变！



# 坑2：[[0]*3]*3 造二维表

# 错误写法：
m = [[0]*3]*3
m[0][0] = 9
print(m)          # [[9,0,0],[9,0,0],[9,0,0]]  ← 三行全变，因为它们本来就是同一个

# 正确写法：
m = [[0]*3 for _ in range(3)]
m[0][0] = 9
print(m)          # [[9,0,0],[0,0,0],[0,0,0]] ✅ 只有第一行变