# Day_02_01_function.py

# 교수님 : f_1 함수를 호출하는 곳의 코드
# 나 : f_1 함수
# 매개변수 : 교수님이 나에게 넘겨주는 데이터
# 반환값 : 내가 교수님에게 넘겨주는 데이터

# 반환값 없고, 매개변수 없고.
def f_1():
    print('f_1')

f_1()

# 반환값 없고, 매개변수 있고.
def f_2(a, b):                  # a, b = 'hello', 'snu'
    print('f_2', a, b, a+b)

def f_22(a1, a2, a3):
    print('f_2', a1, a2, a3)

f_2(12, 34)
f_2('hello', 'snu')
# f_2(12, 'snu')

# 반환값 있고, 매개변수 없고.
def f_3():
    print('f_3')
    return 78

# a = return 78
a = f_3()
print(a)
print(f_3())

if f_3() > 0:
    print('양수')

b = type(78)
print(b)
print(type(78))

# 반환값 있고, 매개변수 있고.
# 문제
# 두 개의 정수를 더하는 함수를 만드세요.
def f_4(a, b):
    # return a+b
    c = a + b
    return c

print(f_4(3, 5))

# print('def f_22(', end='')
# for i in range(10):
#     print('a{}, '.format(i), end='')

# 문제
# 2개의 정수에 대해 큰 수, 작은 수의 순서로 반환하는 함수를 만드세요
# 3 5  -->  3 5
# 5 3  -->  3 5
def order(a, b):
    # if a >= b:
    #     return b, a
    # else:
    #     return a, b

    # if a >= b:
    #     return b, a
    # return a, b

    if a >= b:
        a, b = b, a

    return a, b

c, d = 1, 3
e = 1, 3
f = order(3, 5)
# f = return b, a

print(c, d)
print(e, e[0], e[1])

print(order(3, 5))
print(order(5, 3))

k = order(3, 5)
small, big = order(3, 5)
print(small, big)


# applekoong@naver.com
# 이름을 넣어주세요.



print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')
