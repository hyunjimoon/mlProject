# Day_01_02_if.py

# if : 두 가지 중에서 한 가지 코드만 선택하는 문법
a = 110

if a%2 == 1:
    print('odd')
else:
    print('even')

if a%2:
    print('odd')
else:
    print('even')

if a:
    print('odd')
else:
    print('even')
print('-'*30)

# 문제
# 정수 a가 음수인지 양수인지 0인지 출력하는 코드를 만들어 보세요.
if a > 0:
    print('양수')
else:
    # print('음수, 제로')
    if a < 0:
        print('음수')
    else:
        print('제로')

if a > 0:
    print('양수')
elif a < 0:
    print('음수')
else:
    print('제로')

print('finished.')




print('\n\n\n\n\n\n\n\n\n\n\n\n')
