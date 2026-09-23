# ======================================class==============================
# 类 = 数据 + 动作打包在一起
# __init__ 是"创建对象时自动跑一遍"的函数
# 前后各两个下划线，不是 _init_
# self 就是 C 里你显式传进去的那个结构体指针。
# 方法里调自己的另一个方法，也要 self.

# 完整例子（把"数据 + 动作"绑一起）
class Student:
    def __init__(self,name,score):
        self.name = name
        self.score = score

    def is_pass(self):
        return self.score >= 60

    def __str__(self):
        return self.name + " " + str(self.score)

# 创建对象
s1 = Student("张三",85)
s2 = Student("李四",58)


# 用
print(s1.name)           # 张三      ← 取属性：obj.属性
print(s1.is_pass())      # True      ← 调方法：obj.方法()
print(s2.is_pass())      # False
print(s1)                # 张三 85   ← 有 __str__ 才能这么直接打