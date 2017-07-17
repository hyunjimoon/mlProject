# Day_04_02_MultipleLinear.py
import tensorflow as tf

def not_used():
    x1 = [1, 0, 3, 0, 5]
    x2 = [0, 2, 0, 4, 0]
    y  = [1, 2, 3, 4, 5]

    W1 = tf.Variable(tf.random_uniform([1], -1, 1))
    W2 = tf.Variable(tf.random_uniform([1], -1, 1))
    b  = tf.Variable(tf.random_uniform([1], -1, 1))

    hypothesis = W1*x1 + W2*x2 + b
    cost = tf.reduce_mean(tf.square(hypothesis-y))
    learning_rate = tf.Variable(0.1)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(2001):
        sess.run(train)
        if i % 20 == 0:
            print(sess.run(cost), sess.run(W1), sess.run(W2), sess.run(b))

    sess.close()


def not_used():
    x = [[1., 0., 3., 0., 5.],
         [0., 2., 0., 4., 0.]]
    y = [1., 2., 3., 4., 5.]

    # 여러분이 직접 바꿔 보세요.
    # matmul()

    # [1, 2]
    # [[1, 2]]

    W = tf.Variable(tf.random_uniform([1, 2], -1, 1))
    b = tf.Variable(tf.random_uniform([1], -1, 1))

    hypothesis = tf.matmul(W, x) + b                # (1,2) * (2,5) = (1,5)
    cost = tf.reduce_mean(tf.square(hypothesis-y))
    learning_rate = tf.Variable(0.1)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(2001):
        sess.run(train)
        if i % 20 == 0:
            print(sess.run(cost), sess.run(W), sess.run(b))

    sess.close()

def not_used_3():
    # 문제
    # 행렬에 bias를 넣어 보세요.
    x = [[1., 1., 1., 1., 1.],
         [1., 0., 3., 0., 5.],
         [0., 2., 0., 4., 0.]]
    y = [1., 2., 3., 4., 5.]

    W = tf.Variable(tf.random_uniform([1, 3], -1, 1))

    hypothesis = tf.matmul(W, x)                    # (1,3) * (3,5) = (1,5)
    cost = tf.reduce_mean(tf.square(hypothesis-y))
    learning_rate = tf.Variable(0.1)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(2001):
        sess.run(train)
        if i % 20 == 0:
            print(sess.run(cost), sess.run(W))

    sess.close()


def not_used_4():
    import numpy as np
    trees = np.loadtxt('Data/trees.csv', unpack=True, delimiter=',', skiprows=1)
    print(trees)
    print(trees.shape)

    x = [[1.]*trees.shape[1],
         trees[0],
         trees[1]]
    y = trees[-1]

    W = tf.Variable(tf.random_uniform([1, 3], -1, 1))

    hypothesis = tf.matmul(W, x)                    # (1,3) * (3,5) = (1,5)
    cost = tf.reduce_mean(tf.square(hypothesis-y))
    learning_rate = tf.Variable(0.00015)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(2001):
        sess.run(train)
        if i % 20 == 0:
            print(sess.run(cost), sess.run(W))

    sess.close()

not_used_4()

# import numpy as np
# import matplotlib.pyplot as plt
#
# trees = np.loadtxt('Data/trees.csv', unpack=True, delimiter=',', skiprows=1)
#
# girth, height, volume = trees
# print(girth)
#
# a = (1, 2)
# a1, a2 = (1, 2)
# print(a)
# print(a1, a2)
#
# plt.plot(girth , height, 'ro')
# plt.plot(height, volume, 'go')
# plt.plot(volume, girth , 'bo')

# plt.subplot(221)
# plt.plot(girth , height, 'ro')
#
# plt.subplot(222)
# plt.plot(height, volume, 'go')
#
# plt.subplot(223)
# plt.plot(volume, girth , 'bo')

# from mpl_toolkits.mplot3d import Axes3D
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(girth, height, volume, 'ro')
# plt.show()

# plt.plot(girth , height, 'ro')
# plt.show()
#
# plt.plot(height, volume, 'go')
# plt.show()
#
# plt.plot(volume, girth , 'bo')
# plt.show()
