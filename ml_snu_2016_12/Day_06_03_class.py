# Day_06_03_class.py

class Info:
    def __init__(self, age):
        print('호출됨')
        self.age = age          # 멤버 변수

    def showAge(self, extra):   # 멤버 함수
        print('age :', self.age, extra)

    @property
    def myHope(self):
        return self.age * 100


i1 = Info(50)
i2 = Info(60)
print(i1)
i1.age = 20
print(i1.age)
print(i2.age)

Info.showAge(i1, 11)
Info.showAge(i2, 22)        # unbound method
i2.showAge(33)              # bound method

# print(i1.myHope())
print(i1.myHope)











