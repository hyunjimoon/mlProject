# Day_05_05_Logistic_iris.py
import tensorflow as tf
import numpy as np
import csv

def getIrisLogistic(filename, speciesTrue, speciesFalse):
    with open(filename, 'r', encoding='utf-8') as f:

        # skip header.
        f.readline()

        rows = []
        # for row in f:
        for row in csv.reader(f):

            if row[-1] != speciesTrue and row[-1] != speciesFalse:
                continue

            # print(int(row[-1] == speciesTrue))
            # print(row)
            # line = row[1:-1]
            line = [1.]
            line += [float(i) for i in row[1:-1]]    # comprehension
            line.append(int(row[-1] == speciesTrue))
            # line = []
            # for i in row[1:-1]:
            #     line.append(float(i))
            # print(line)
            rows.append(line)

        train = rows[10:-10]
        test  = rows[:10] + rows[-10:]

        # (80,6) --> (6,80)
        return np.transpose(train), np.transpose(test)


def showAccuracy(train_set, test_set):
    x = train_set[:-1]
    y = train_set[-1]

    X = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)

    # len(x)  --> x.shape[0]
    W = tf.Variable(tf.random_uniform([1, len(x)], -1, 1))

    z = tf.matmul(W, X)
    hypo = tf.div(1., 1. + tf.exp(-z))
    cost = -tf.reduce_mean(Y * tf.log(hypo) + (1 - Y) * tf.log(1 - hypo))

    rate = tf.Variable(0.1)
    train = tf.train.GradientDescentOptimizer(rate).minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(2001):
        sess.run(train, feed_dict={X: x, Y: y})
        # if i % 20 == 0:
        #     print(sess.run(cost, feed_dict={X: x, Y: y}),
        #           sess.run(W))

    # ------------------------------------------------ #

    # 2차원
    y_hats = sess.run(hypo, feed_dict={X: test_set[:-1]})
    # print(y_hats)
    # print(y_hats > 0.5)

    # 1차원
    y_hats = (y_hats[0] > 0.5)
    print(y_hats)
    print(y_hats == test_set[-1])
    print(np.mean(y_hats == test_set[-1]))

    sess.close()


train_set, test_set = getIrisLogistic('Data/iris.csv', 'setosa', 'versicolor')
showAccuracy(train_set, test_set)
# print(train_set.shape)
# print( test_set.shape)

train_set, test_set = getIrisLogistic('Data/iris.csv', 'versicolor', 'virginica')
showAccuracy(train_set, test_set)

train_set, test_set = getIrisLogistic('Data/iris.csv', 'virginica', 'setosa')
showAccuracy(train_set, test_set)

# iris 데이터셋을 사용해서 아까 만든 코드에 접목시켜 보세요.






