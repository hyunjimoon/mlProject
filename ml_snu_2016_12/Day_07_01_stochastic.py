# Day_07_01_stochastic.py
import numpy as np
import matplotlib.pyplot as plt
import random

def loadAction():
    # xy = np.loadtxt('Data/action.txt', unpack=True, delimiter=',')
    #
    # x = xy[:-1]
    # y = xy[-1]
    # x = x.transpose()
    # y = y.transpose()
    #
    # print(x.shape)      # (100, 3)
    # print(y.shape)      # (100,)

    xy = np.loadtxt('Data/action.txt', delimiter=',')
    # print(xy.shape)     # (100, 4)

    x = xy[:, :-1]
    y = xy[:, -1:]

    # print(x.shape)      # (100, 3)
    # print(y.shape)      # (100,)

    return x, y

def sigmoid(z):
    return 1./(1.+np.exp(-z))

def gradDescent(x, y):
    m, n = x.shape              # 100, 3
    alpha = 0.01
    weight = np.ones((n, 1))    # (3, 1)

    for _ in range(m):
        h = sigmoid(np.dot(x, weight))                  # (100, 1) = (100, 3) * (3, 1)
        # error = y - h                                   # (100, 1) = (100, 1) - (100, 1)
        # grad  = alpha * np.dot(x.transpose(), error)    # (3, 1) = (3, 100) * (100, 1)
        # weight= weight + grad                           # (3, 1) = (3, 1) + (3, 1)
        # print(weight)

        error = h - y                                   # (100, 1) = (100, 1) - (100, 1)
        grad  = alpha * np.dot(x.transpose(), error)    # (3, 1) = (3, 100) * (100, 1)
        weight= weight - grad                           # (3, 1) = (3, 1) + (3, 1)
        print(weight)

    return weight


def stocGradDescent(x, y):
    m, n = x.shape          # 100, 3
    alpha = 0.01
    weight = np.ones(n)     # 3

    for i in range(m*10):
        pos = i%m
        h = sigmoid(np.sum(x[pos]*weight))
        error = h - y[pos]
        grad  = alpha * error * x[pos]
        weight= weight - grad

    return weight


def stocGradDescentUpgrade(x, y):
    m, n = x.shape          # 100, 3
    alpha = 0.01
    weight = np.ones(n)     # 3

    for i in range(m*10):
        # pos = random.randrange(m)
        pos = int(random.uniform(0, m))
        h = sigmoid(np.sum(x[pos]*weight))
        error = h - y[pos]
        grad  = alpha * error * x[pos]
        weight= weight - grad

    return weight


def bestFit(x, y, w):
    xx1, yy1 = [], []
    xx2, yy2 = [], []

    for i in range(x.shape[0]):
        x1, x2 = x[i,1], x[i,2]

        if int(y[i,0]) == 1:
            xx1.append(x1)
            yy1.append(x2)
        else:
            xx2.append(x1)
            yy2.append(x2)

    plt.plot(xx1, yy1, 'ro')
    plt.plot(xx2, yy2, 'go')

    xx = np.arange(-3, 3, 0.1)
    yy = -(w[0] + w[1]*xx) / w[2]
    plt.plot(xx, yy, 'b--')

    plt.show()

    # 0 = w[0] + w[1]*x1 + w[2]*x2
    # -w[2]*x2 = w[0] + w[1]*x1
    # x2 = (w[0] + w[1]*x1) / -w[2]
    # x2 = -(w[0] + w[1]*x1) / w[2]


x, y = loadAction()
# w = gradDescent(x, y)
# w = stocGradDescent(x, y)
w = stocGradDescentUpgrade(x, y)
bestFit(x, y, w)













