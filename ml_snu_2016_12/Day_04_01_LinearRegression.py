# Day_04_01_LinearRegression.py
import tensorflow as tf

def not_used():
    x = [1, 2, 3]
    y = [1, 2, 3]

    W = tf.Variable(tf.random_uniform([1], -1, 1))
    b = tf.Variable(tf.random_uniform([1], -1, 1))

    hypothesis = W*x + b     # tf.add(tf.mul(W, x), b)
    cost = tf.reduce_mean((hypothesis-y) ** 2)
    # cost = tf.reduce_mean(tf.square(hypothesis-y))
    learning_rate = tf.Variable(0.1)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(2001):
        # result = sess.run(train)
        # print(result)
        sess.run(train)

        # print(type(sess.run(W)), sess.run(W)[0])

        if i%20 == 0:
            print(sess.run(cost), sess.run(W), sess.run(b))

    sess.close()

def not_used_2():
    # 문제
    # 위의 코드를 placeholder 버전으로 변경해 보세요.
    # 7과 11에 대해서 예측을 해보세요.
    x = [1, 2, 3]
    y = [1, 2, 3]

    X = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)

    W = tf.Variable(tf.random_uniform([1], -1, 1))
    b = tf.Variable(tf.random_uniform([1], -1, 1))

    hypothesis = W*X + b
    cost = tf.reduce_mean(tf.square(hypothesis-Y))
    learning_rate = tf.Variable(0.1)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(2001):
        sess.run(train, feed_dict={X: x, Y: y})

    print(sess.run(hypothesis, feed_dict={X:  7}))
    print(sess.run(hypothesis, feed_dict={X: 11}))
    print(sess.run(hypothesis, feed_dict={X: [7, 11]}))

    sess.close()

import numpy as np

# xy = np.loadtxt('Data/simple.txt', unpack=True)
# xy = np.loadtxt('Data/simple2.txt', unpack=True, delimiter=',', skiprows=1)
# print(xy)
# print(type(xy))
# print(xy[0], xy[1])

xy = np.loadtxt('Data/cars.csv', unpack=True, delimiter=',', skiprows=1)

x = xy[0]       # [1, 2, 3]
y = xy[1]       # [1, 2, 3]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_uniform([1], -1, 1))
b = tf.Variable(tf.random_uniform([1], -1, 1))

hypothesis = W*X + b
cost = tf.reduce_mean(tf.square(hypothesis-Y))
learning_rate = tf.Variable(0.0035)

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(2001):
    sess.run(train, feed_dict={X: x, Y: y})
    if i%20 == 0:
        print(sess.run(cost, feed_dict={X: x, Y: y}))

# print(sess.run(hypothesis, feed_dict={X: [30, 50]}))

WW = sess.run(W)
bb = sess.run(b)

sess.close()

# 문제
# cars.csv 파일을 이용해서
# 속도가 30과 50일 때의 제동거리를 구해보세요.
import matplotlib.pyplot as plt

def prediction(x, W, b):
    return W*x + b

plt.plot(x, y, 'ro')
plt.plot((0, 25), (0, prediction(25, WW, bb)))
plt.plot((0, 25), (prediction(0, WW, bb), prediction(25, WW, bb)))
plt.show()






