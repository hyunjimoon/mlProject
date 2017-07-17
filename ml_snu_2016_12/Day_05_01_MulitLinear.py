# Day_05_01_MulitLinear.py
import tensorflow as tf
import numpy as np


def showResult(x1, x2, y, rate, loop_count):
    # x = [[1.]*len(x1), x1, x2]
    # y = trees[-1]
    ones = np.ones((len(x1)), dtype=np.float32)
    x = np.vstack((ones, x1, x2))

    W = tf.Variable(tf.random_uniform([1, 3], -1, 1))

    hypothesis = tf.matmul(W, x)  # (1,3) * (3,5) = (1,5)
    cost = tf.reduce_mean(tf.square(hypothesis - y))
    learning_rate = tf.Variable(rate)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(loop_count):
        sess.run(train)
        if i % 20 == 0:
            print(sess.run(cost), sess.run(W))

    sess.close()

def showPlaceHolder(x1, x2, y, rate, loop_count):
    ones = np.ones((len(x1)), dtype=np.float32)
    x = np.vstack((ones, x1, x2))
    # print(x.shape)
    # print(y.shape)
    # print(x)

    X = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)

    W = tf.Variable(tf.random_uniform([1, 3], -1, 1))

    hypothesis = tf.matmul(W, X)  # (1,3) * (3,5) = (1,5)
    cost = tf.reduce_mean(tf.square(hypothesis - Y))
    learning_rate = tf.Variable(rate)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(loop_count):
        sess.run(train, feed_dict={X: x, Y: y})

    d1 = [[1.], [8.8], [63]]    # 10.2
    print(sess.run(hypothesis, feed_dict={X: d1}))

    d2 = [[1., 1.], [8.8, 10.5], [63, 72]]    # 10.2, 16.4
    print(sess.run(hypothesis, feed_dict={X: d2}))

    sess.close()


trees = np.loadtxt('Data/trees.csv',
                   unpack=True, delimiter=',',
                   skiprows=1, dtype=np.float32)
# print(trees)
# print(trees.shape)

# showResult(trees[0], trees[1], trees[2], 0.00015, 2000)
# showResult(trees[1], trees[2], trees[0], 0.0001, 1000)
# showResult(trees[2], trees[0], trees[1], 0.0007, 1500)

# 문제"Girth","Height","Volume"
# "Girth","Height","Volume"
#    0       1        2
# x1, x2, y  -->  "Height","Volume","Girth"
# x1, x2, y  -->  "Volume","Girth","Height"

showPlaceHolder(trees[0], trees[1], trees[2], 0.00015, 2000)

# 문제
# 아래 데이터를 예측하는 placeholder 버전으로 업그레이드 해보세요.
# trees.csv 파일에서 3번째와 4번째 데이터를 가져왔습니다.

# x1  x2  y
# 8.8,63,10.2
# 10.5,72,16.4
