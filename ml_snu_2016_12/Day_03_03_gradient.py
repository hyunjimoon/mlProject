# Day_03_03_gradient.py
import matplotlib.pyplot as plt

# x로 미분
# y = x     --> 1
# y = 2x    --> 2
# y = 3     --> 0
# y = x^2   --> 2x

def gradient_descent(x, y, W):
    grad = 0
    for i in range(len(x)):
        grad += (W*x[i] - y[i]) * x[i]
    return grad / len(x)

x = [1, 2, 3]
y = [1, 2, 3]

W = 100
for i in range(10):
    grad = gradient_descent(x, y, W)
    W = W - grad*0.1

    print(W)
    plt.plot((0, 5), (0, 5*W))

plt.plot(x, y, 'ro')
plt.xlim(0, 5)          # plt.xlim((0, 5))
plt.ylim(0, 5)
plt.show()





