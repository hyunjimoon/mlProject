# Day_02_03_cost.py
import matplotlib.pyplot as plt

def cost(x, y, W):
    loss = 0
    for i in range(len(x)):
        hx = W * x[i]
        loss += (hx - y[i]) ** 2
    return loss / len(x)

# H(x) = Wx + b   --> W=1, b=0
x = [1, 2, 3]
y = [1, 2, 3]

xx, yy = [], []
for i in range(-30, 50):
    # print(i/10)
    W = i / 10          # -3 ~ 4.9
    c = cost(x, y, W)

    print(W, c)

    xx.append(W)
    yy.append(c)

# plt.plot(x, y)
plt.plot(xx, yy, 'ro')
plt.show()





