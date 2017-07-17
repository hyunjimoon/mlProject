# Day_01_03_for.py

# 1 3 5 7 9         1, 9, 2         시작, 종료, 증감
# 2 4 6 8 10        2, 10, 2
# 5 4 3 2 1         5, 1, -1

for i in range(1, 11, 2):
    print(i, end=' ')
print()

for i in range(2, 12, 2):
    print(i, end=' ')
print()

for i in range(5, 0, -1):
    # print(i, end=' ')
    print('hello')
print()

print(12, 34, 56, sep='**')

for i in range(0, 5, 1):    # 시작, 종료, 증감
    print(i, end=' ')
print()

for i in range(0, 5):       # 시작, 종료, 증감(1)
    print(i, end=' ')
print()

for i in range(5):          # 시작(0), 종료, 증감(1)
    print(i, end=' ')
print()

for i in range(4, -1, -1):
    print(i, end=' ')
print()

for i in reversed(range(5)):
    print(i, end=' ')
print()

# 1, 9, 2
i = 1               # 시작
while i < 11:       # 종료
    print(i, end=' ')
    i = i+2         # 증감
print()

i = 1               # 시작
while i <= 9:       # 종료
    print(i, end=' ')
    i += 2          # 증감
print()

# s = 0
# while True:
#     print('never stop.')
#     n = int(input('number :'))
#     if n < 0:
#         break
#     s += n
#
# print(s)

# 문제
# 입력 받은 정수 중에서 음수와 양수의 합계를 각각 구해 보세요.
# 0을 입력할 때까지 반복.
# 3 9 -7 -2 4 0
# 16 -9
# pos, neg = 0, 0
# while True:
#     n = int(input('number : '))
#     if n == 0:
#         break
#     if n > 0:
#         pos += n
#     else:
#         neg += n
#
# print(pos, neg)

# exception
# n = input('number : ')
# print(n)

# 문제
# 0부터 99까지의 정수를 한 줄에 10개씩 출력해 보세요.
# for문 사용합니다.
for i in range(100):
    print(i, end=' ')

    if i%10 == 9:
        print()

#  0  1  2  3  4  5  6  7  8  9
# 10 11 12 13 14 15 16 17 18 19
# 20 21 22 23 24 25 26 27 28 29
# 30 31 32 33 34 35 36 37 38 39
# 40 41 42 43 44 45 46 47 48 49
# 50 51 52 53 54 55 56 57 58 59
# 60 61 62 63 64 65 66 67 68 69
# 70 71 72 73 74 75 76 77 78 79
# 80 81 82 83 84 85 86 87 88 89
# 90 91 92 93 94 95 96 97 98 99

for i in reversed(range(100)):
    print(i, end=' ')

    if i%10 == 0:
        print()
print('-'*30)

print(12, 3.14, 'hello', sep=',')
# base = '{},' * 3
# t = base.format(12, 3.14, 'hello')
t = '{1},{2},{0}'.format(12, 3.14, 'hello')
t = '{1},{2},{0},{0}'.format(12, 3.14, 'hello')
t = '{0:10},{1:10},{2:10}'.format(12, 3.14, 'hello')
t = '{:<10},{:>10.5f},{:^10}'.format(12, 3.14, 'hello')
print(t)


print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n')









