class Student:
    def __init__(self,name,score):
        self.name = name
        self.score = score

    def is_pass(self):
        return self.score >= 60

    def __str__(self):
        return self.name + " " + str(self.score)

students = [Student("张三",85),Student("李四",58),Student("王五",60)]      # 实例化List

for st in students:
    print(st.__str__() + " " + str(st.is_pass()))