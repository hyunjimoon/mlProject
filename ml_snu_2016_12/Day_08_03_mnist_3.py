# Day_08_03_mnist_3.py
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import math

mnist = input_data.read_data_sets('Data/mnist/', one_hot=True)
print(type(mnist))          # train, validation, test
print(type(mnist.train))

learning_rate = 0.1
training_epoches = 25
batch_size = 100
display_step= 1

x = tf.placeholder(tf.float32, [None, 784])     # 28x28 = 784
y = tf.placeholder(tf.float32, [None,  10])

# --------------------------------------------------- #

learning_rate = 0.01
training_epoches = 15

# W1 = tf.get_variable('W1', shape=[784, 256],
#                      initializer=tf.contrib.layers.xavier_initializer())
# W2 = tf.get_variable('W2', shape=[256, 256],
#                      initializer=tf.contrib.layers.xavier_initializer())
# W3 = tf.get_variable('W3', shape=[256,  10],
#                      initializer=tf.contrib.layers.xavier_initializer())
#
# B1 = tf.Variable(tf.zeros([256]))
# B2 = tf.Variable(tf.zeros([256]))
# B3 = tf.Variable(tf.zeros([ 10]))
#
# L1 = tf.nn.relu(tf.add(tf.matmul(x, W1), B1))
# L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), B2))
#
# activation = tf.add(tf.matmul(L2, W3), B3)

W1 = tf.get_variable('W1', shape=[784, 256],
                     initializer=tf.contrib.layers.xavier_initializer())
W2 = tf.get_variable('W2', shape=[256, 256],
                     initializer=tf.contrib.layers.xavier_initializer())
W3 = tf.get_variable('W3', shape=[256, 256],
                     initializer=tf.contrib.layers.xavier_initializer())
W4 = tf.get_variable('W4', shape=[256, 256],
                     initializer=tf.contrib.layers.xavier_initializer())
W5 = tf.get_variable('W5', shape=[256,  10],
                     initializer=tf.contrib.layers.xavier_initializer())

B1 = tf.Variable(tf.zeros([256]))
B2 = tf.Variable(tf.zeros([256]))
B3 = tf.Variable(tf.zeros([256]))
B4 = tf.Variable(tf.zeros([256]))
B5 = tf.Variable(tf.zeros([ 10]))

dropout_rate = tf.placeholder(tf.float32)

_L1 = tf.nn.relu(tf.add(tf.matmul(x, W1), B1))
L1  = tf.nn.dropout(_L1, dropout_rate)
_L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), B2))
L2  = tf.nn.dropout(_L2, dropout_rate)
_L3 = tf.nn.relu(tf.add(tf.matmul(L2, W3), B3))
L3  = tf.nn.dropout(_L3, dropout_rate)
_L4 = tf.nn.relu(tf.add(tf.matmul(L3, W4), B4))
L4  = tf.nn.dropout(_L4, dropout_rate)

activation = tf.add(tf.matmul(L4, W5), B5)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(activation, y))

# --------------------------------------------------- #

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epoches):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)  # 55000/100 = 550

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)

            # --------------------------------------------------- #
            _, c = sess.run([optimizer, cost],
                            feed_dict={x: batch_xs, y: batch_ys, dropout_rate: 0.7})
            # --------------------------------------------------- #

            # sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            # c = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})

            avg_cost += c / total_batch

        if (epoch+1)%display_step == 0:
            print('epoch: {}, cost: {}'.format(epoch+1, avg_cost))

    # ---------------------------------------- #

    pred = tf.equal(tf.argmax(activation, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(pred, tf.float32))

    # --------------------------------------------------- #
    print('accuracy :',
          sess.run(accuracy, {x: mnist.test.images, y: mnist.test.labels, dropout_rate: 1.0}))
    # print('accuracy :', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
    # --------------------------------------------------- #

# 결과
# epoch: 13, cost: 0.24250536575913414
# epoch: 14, cost: 0.23337515430694278
# epoch: 15, cost: 0.22478871546685686
# accuracy : 0.9575

