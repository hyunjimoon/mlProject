# Day_07_02_mnist_simple.py
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('Data/mnist/', one_hot=True)
print(type(mnist))          # train, validation, test
print(type(mnist.train))

learning_rate = 0.1
training_epoches = 25
batch_size = 100
display_step= 1

x = tf.placeholder(tf.float32, [None, 784])     # 28x28 = 784
y = tf.placeholder(tf.float32, [None,  10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

activation = tf.nn.softmax(tf.matmul(x, W) + b)     # (?, 784) * (784, 10) = (?, 10)

cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(activation), reduction_indices=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# sess = tf.Session()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epoches):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)  # 55000/100 = 550

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)

            _, c = sess.run([optimizer, cost],
                            feed_dict={x: batch_xs, y: batch_ys})

            # sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            # c = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})

            avg_cost += c / total_batch

        if (epoch+1)%display_step == 0:
            print('epoch: {}, cost: {}'.format(epoch+1, avg_cost))

    # ---------------------------------------- #

    import random
    r = random.randrange(mnist.test.num_examples)
    print('label :', sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
    print('prediction: ',
          sess.run(tf.argmax(activation, 1), {x: mnist.test.images[r:r+1]}))

    import matplotlib.pyplot as plt
    plt.imshow(mnist.test.images[r:r+1].reshape(28, 28),
               cmap='Greys', interpolation='nearest')
    plt.show()

    pred = tf.equal(tf.argmax(activation, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(pred, tf.float32))

    print('accuracy :',
          sess.run(accuracy, {x: mnist.test.images, y: mnist.test.labels}))
    # print('accuracy :', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

# sess.close()

# 결과
# epoch: 23, cost: 0.2678563750738446
# epoch: 24, cost: 0.2668886929614977
# epoch: 25, cost: 0.26601178973913203
# label : [4]
# prediction:  [4]
# accuracy : 0.923








