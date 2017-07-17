# Day_01_01_Basic.py
# import tensorflow as tf
# 최초 : ctrl + shift + f10
# 다음 : shift + f10

print('Hello, "TensorFlow!"')
print("Hello, 'TensorFlow!'")
print(12, 3.14, 'Hello', True)
print(type(12), type(3.14), type('Hello'), type(True))

a = 12
b = 3.14
a, b = 12, 3.14
# a = 12, b = 3.14
# 12 = a
print(a, b)

c = 12, 3.14        # () -> tuple
print(c)
print(c[0], c[1])
print('-' * 30)

# 연산자(operator) : 산술, 관계, 논리
# 산술 : +  -  *  /  //  **  %
a, b = 13, 6
print(a +  b)
print(a -  b)
print(a *  b)
print(a /  b)
print(a // b)
print(a ** b)   # 지수, exponent
print(a %  b)

#      2        //
#   +----
# 6 | 13
#     12
#    ----
#      1        %

# 문제
# 두 자리 양수를 거꾸로 뒤집어 보세요.
# 29  -->  92
# 2*10 + 9
# 9*10 + 2
n = 29
# m1 = n//10
# m2 = n %10
# n = m2*10 + m1
n = n%10*10 + n//10
print(n)

#       2
#    +----
# 10 | 29
#      20
#     ----
#       9
print('-'*30)

# 관계 : >  >=  <  <=  ==  !=
print(a, b)     # 13, 6

print(a >  b)
print(a >= b)
print(a <  b)
print(a <= b)
print(a == b)
print(a != b)

# age = int(input('number : '))
# print(type(age))
# print(10 <= age <= 19)

# 논리 : and  or  not
# 다른 언어 : &&  ||  !

print(True  and True )
print(True  and False)
print(False and True )
print(False and False)

# 비트 : &  |  ^  ~











print('\n\n\n\n\n\n\n\n\n\n\n\n\n')