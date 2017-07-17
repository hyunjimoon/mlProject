# Day_06_01_softmax.py

def showSoftMax():
    import math

    a1 = math.e ** 2.0
    a2 = math.e ** 1.0
    a3 = math.e ** 0.1

    base = a1 + a2 + a3

    print(a1/base)  # 0.7   <--  0.6590011388859678
    print(a2/base)  # 0.2   <--  0.2424329707047139
    print(a3/base)  # 0.1   <--  0.09856589040931818


import tensorflow as tf
import numpy as np

xy = np.loadtxt('Data/05train.txt',
                unpack=True, dtype=np.float32)
print(xy)
print(xy.shape)     # (6, 8)

# x = xy[:3]
# y = xy[-3:]
#
# print(x.shape)    # (3,8)
# print(y.shape)    # (3,8)
# print(x)
# print(y)

x = xy[:3].transpose()
y = xy[-3:].transpose()

print(x.shape)      # (8,3)
print(y.shape)      # (8,3)
print(x)
print(y)

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.zeros([3, 3]))

z = tf.matmul(X, W)     # (8,3) x (3,3) = (8,3)
hypo = tf.nn.softmax(z)
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypo),
                                     reduction_indices=1))  # 1=row
rate = tf.Variable(0.1)
train= tf.train.GradientDescentOptimizer(rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(2001):
    sess.run(train, feed_dict={X: x, Y: y})

    if i%20 == 0:
        print(i, sess.run(cost, feed_dict={X: x, Y: y}))
        print(sess.run(W))

# 11시간 공부하고 7번 수업 참석에 대해서 학점 예측
#  3시간 공부하고 4번 수업 참석에 대해서 학점 예측
# [[ 1.  2.  1.]
#  [ 1.  3.  2.]
#  [ 1.  3.  4.]
#  [ 1.  5.  5.]
#  [ 1.  7.  5.]
#  [ 1.  2.  5.]
#  [ 1.  6.  6.]
#  [ 1.  7.  7.]]

a = sess.run(hypo, feed_dict={X: [[1, 11, 7]]})
b = sess.run(hypo, feed_dict={X: [[1,  3, 4]]})
c = sess.run(hypo, feed_dict={X: [[1, 11, 7], [1,  3, 4]]})
print(a)    # [[  8.78134191e-01   1.21488057e-01   3.77686898e-04]]
print(b)    # [[ 0.04476492  0.44394752  0.51128757]]
print(c)    # [[  8.78134191e-01   1.21488057e-01   3.77686898e-04]
            #  [  4.47649248e-02   4.43947524e-01   5.11287570e-01]]


print(sess.run(tf.argmax(a, 1)))    # [0]
print(sess.run(tf.argmax(b, 1)))    # [2]
print(sess.run(tf.argmax(c, 1)))    # [0 2]
print(sess.run(tf.argmax(c, 0)))    # [0 1 1]

g1 = sess.run(tf.argmax(a, 1))
g2 = sess.run(tf.argmax(b, 1))
g3 = sess.run(tf.argmax(c, 1))

nomial = ['A', 'B', 'C']
print(nomial[g1[0]])                    # A
print(nomial[g2[0]])                    # C
print(nomial[g3[0]], nomial[g3[1]])     # A C
# print(nomial[g3])

sess.close()

