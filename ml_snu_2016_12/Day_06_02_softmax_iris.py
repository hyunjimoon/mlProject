# Day_06_02_softmax_iris.py
import csv
import numpy as np
import tensorflow as tf
import random

def makeNewIris():
    def makeOneHot(species):
        if species == 'setosa':     return [1, 0, 0]
        if species == 'versicolor': return [0, 1, 0]
        if species == 'virginica':  return [0, 0, 1]
        return None

    with open('Data/iris.csv', 'r', encoding='utf-8') as f:
        f.readline()

        rows = []
        for row in csv.reader(f):
            # print(row)
            line = [1.]
            line += [float(i) for i in row[1:-1]]
            line.extend(makeOneHot(row[-1]))
            # print(line)
            rows.append(line)

        random.shuffle(rows)

        # -------------------------------- #

        with open('Data/iris_softmax.csv', 'w', encoding='utf-8', newline='') as fw:
            csv.writer(fw).writerows(rows)
            # writer = csv.writer(fw)
            # for row in rows:
            #     writer.writerow(row)


def readIrisSoftMax(index):
    with open('Data/iris_softmax.csv', 'r', encoding='utf-8') as f:
        start = index * 30
        end   = start + 30

        x_train, y_train, x_test, y_test = [], [], [], []
        for i, row in enumerate(csv.reader(f)):
            if start <= i < end:
                x_test.append(row[:5])
                y_test.append(row[5:])
            else:
                x_train.append(row[:5])
                y_train.append(row[5:])

        return np.array(x_train, dtype=np.float32), \
               np.array(y_train, dtype=np.float32), \
               np.array(x_test , dtype=np.float32), \
               np.array(y_test , dtype=np.float32)

def showAccuracy(x_train, y_train, x_test, y_test):
    x = x_train
    y = y_train

    X = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)

    W = tf.Variable(tf.zeros([5, 3]))

    z = tf.matmul(X, W)  # (120,5) x (5,3) = (120,3)
    hypo = tf.nn.softmax(z)
    cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypo),
                                         reduction_indices=1))  # 1=row
    rate = tf.Variable(0.1)
    train = tf.train.GradientDescentOptimizer(rate).minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(2001):
        sess.run(train, feed_dict={X: x, Y: y})
        # if i % 20 == 0:
        #     print(i, sess.run(cost, feed_dict={X: x, Y: y}))
        #     print(sess.run(W))

    # ------------------------------------- #

    r1 = sess.run(hypo, feed_dict={X: x_test})
    r2 = sess.run(tf.argmax(r1, 1))

    # print(r1)
    # print(r2)

    y1 = sess.run(tf.argmax(y_test, 1))
    # print(y1)
    # print(r2 == y1)
    print(np.mean(r2 == y1))

# makeNewIris()

for i in range(5):
    # x_train, y_train, x_test, y_test = readIrisSoftMax(i)
    # showAccuracy(x_train, y_train, x_test, y_test)
    showAccuracy(*readIrisSoftMax(i))

print('\n\n\n\n\n\n\n\n\n\n\n')

# print( [1, 3, 5])
# print(*[1, 3, 5])

# t = readIrisSoftMax(0)
# print(t)
# print(type(t))      # tuple

# x_train, y_train, x_test, y_test = readIrisSoftMax(0)
#
# print(x_train.shape)    # (120, 5)
# print(y_train.shape)    # (120, 3)
# print(x_test .shape)    # (30, 5)
# print(y_test .shape)    # (30, 3)

# 문제
# x_test를 사용해서 예측해 보세요.






