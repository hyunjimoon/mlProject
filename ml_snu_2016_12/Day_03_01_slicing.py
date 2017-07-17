# Day_03_01_slicing.py

# for i in range(10):
#     print(i)

a = list(range(10))
print(a)
print(a[0], a[-1], len(a))
print(a[0:3])
print(a[0:len(a)//2])
print(a[len(a)//2:len(a)])
print(a[:3])
print(a[:len(a)//2])
print(a[len(a)//2:])
print(a[:])
print(a[::2])       # 종료는 포함되지 않음.
print(a[1::2])      # 시작:종료:증감
print(a[len(a)-1:-1:-1])
print(a[-1:-1:-1])
print(a[3:3])
print(a[-1::-1])
print(a[::-1])
print(a[::-2])
print(a[-2::-2])

# 문제
# 앞쪽 절반을 슬라이싱으로 출력해 보세요.
# 뒤쪽 절반을 슬라이싱으로 출력해 보세요.
# 짝수 번째만 출력해 보세요.
# 홀수 번째만 출력해 보세요.
# 거꾸로 출력해 보세요.
# 거꾸로 짝수 번째만 출력해 보세요.
# 거꾸로 홀수 번째만 출력해 보세요.
print('-'*30)

b = a
c = a[:]
a[0] = -100
print(a)
print(b)
print(c)

for i in a[::2]:
    print(i)

for i in range(0, len(a), 2):
    print(a[i])
