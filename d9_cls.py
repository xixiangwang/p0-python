# Rectangle 类（自己敲，别抄 d8_student.py）：

# __init__ 收 width / height → 存成 self.width / self.height
# area(self) → 返回面积
# is_square(self) → 宽高相等返回 True
# __str__(self) → 返回 "矩形 3x4" 这种

class Rectangle():
    def __init__(self,width,height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

    def is_square(self):
        return self.width == self.height

    def __str__(self):
        return "矩形" + " " +  str(self.width) + "*" + str(self.height)

r1 = Rectangle(3, 4)
r2 = Rectangle(5, 5)
print(r1.area())        # 12
print(r2.area())        # 25
print(r1.is_square())   # False
print(r2.is_square())   # True
print(r1)               # 矩形 3x4
