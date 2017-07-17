import tensorflow as tf
import numpy as np

def not_used():
    xy = np.loadtxt('Data/trees.csv',
                    unpack=True, delimiter=',', skiprows=1, dtype=np.float32)
    print(xy)
    print(xy.shape)
    # x = [xy[0],
    #      xy[1]]
    # TypeError: Input 'b' of 'MatMul' Op has type float64 that does not match type float32 of argument 'a'.
    x = xy[:2]
    # x1 = xy[0]
    # x2=xy[1]
    y=xy[-1]
    print(x)

    # xx = [[1.,1.,1.,1.,1.],
    #     [1., 0., 3., 0., 5.],
    #      [0., 2., 0., 4., 5.]]
    # y = [1., 2., 3., 4., 5.]

    print(type(xy))
    print(xy.dtype)

    W = tf.Variable(tf.random_uniform([1, 2], -1, 1))
    b = tf.Variable(tf.random_uniform([2,1], -1, 1))
    # W1 = tf.Variable(tf.random_uniform([1], -1, 1))
    # W2 = tf.Variable(tf.random_uniform([1], -1, 1))

    hypothesis = tf.matmul(W, x) + b
    cost = tf.reduce_mean((hypothesis - y) ** 2)
    learning_rate = tf.Variable(0.0001)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(2001):
        sess.run(train)

    sess.close()


def not_used_2():
    trees = np.loadtxt('Data/trees.csv',
                       unpack=True, delimiter=',',
                       skiprows=1, dtype=np.float32)
    print(trees)
    print(trees.shape)  # 데이터 크기 표현
    x = trees[:2]
    y = trees[-1]  # 마지막 열을 표현. trees[2]를 쓰려면 마지막 줄이 몇번째인지 알아야하므로

    # tf.matmul()
    # [1,2] : 1차원 : 데이터가 2개
    # [[1,2]] : 2차원 : 데이터가 1개 , 그 데이터 1개 안에 들어가보니 데이터가 2개씩 들어있다. 이것이 1x2 행렬

    W = tf.Variable(tf.random_uniform([1, 2], -1, 1))  # 1x2 행렬로 만들어줘야함 [2]는 그냥 단순히 1차원
    b = tf.Variable(tf.random_uniform([1], -1, 1))

    hypothesis = tf.matmul(W, x) + b
    cost = tf.reduce_mean(tf.square(hypothesis - y))
    learning_rate = tf.Variable(0.00015)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(2001):

        sess.run(train)
        if i % 20 == 0:  # 20번에 한번만 출력
            print(sess.run(cost), sess.run(W), sess.run(b))

    sess.close()


not_used_2()
