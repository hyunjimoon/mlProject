# Day_02_02_list.py

a = [1, 3, 5, 7]
print(a)
print(a[0], a[1], a[2])

print('hello', len('hello'))

for i in range(len(a)):
    print(i, a[i])
print('-'*30)

# 문제
# a를 거꾸로 출력해 보세요.
# 3 2 1 0   -->  3, 0, -1
for i in range(len(a)-1, -1, -1):
    print(i, a[i])

for i in reversed(range(len(a))):
    print(i, a[i])

for i in a:     # range(), list --> iterable(반복할 수 있는 객체)
    print(i)

for i in reversed(a):
    print(i)

print(type(range(5)), type(a))
print('-'*30)

print(a)

a[0] = 99
a[1] = a[0]
print(a)
print(a[-1], a[-2])
print(a[len(a)-1], a[len(a)-2])
print('-'*30)

b = (1, 3, 5)
print(b)
print(b[0], b[1], b[2])
print(b[-1], b[-2])

# b[0] = 99

# []    ()     {}              <>
# list  tuple  set/dictionary  not_used
b = a
print(b)
print('-'*30)

c1 = [2, 5, 9]
c1[-1] = -1
c2 = c1
print(c1)
print(c2)

import random
print(random.randrange(10))
print(random.randrange(0, 10))
print(random.randrange(0, 10, 2))

for i in range(10):
    print(random.randrange(5), end=' ')

# 문제









print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n')

