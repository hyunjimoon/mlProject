# Day_08_02_mnist_2.py
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import math

mnist = input_data.read_data_sets('Data/mnist/', one_hot=True)
print(type(mnist))          # train, validation, test
print(type(mnist.train))

def xavier_init(n_inputs, n_outputs, uniform=True):
  """Set the parameter initialization using the method described.
  This method is designed to keep the scale of the gradients roughly the same
  in all layers.
  Xavier Glorot and Yoshua Bengio (2010):
           Understanding the difficulty of training deep feedforward neural
           networks. International conference on artificial intelligence and
           statistics.
  Args:
    n_inputs: The number of input nodes into each output.
    n_outputs: The number of output nodes for each input.
    uniform: If true use a uniform distribution, otherwise use a normal.
  Returns:
    An initializer.
  """
  if uniform:
    # 6 was used in the paper.
    init_range = math.sqrt(6.0 / (n_inputs + n_outputs))
    return tf.random_uniform_initializer(-init_range, init_range)
  else:
    # 3 gives us approximately the same limits as above since this repicks
    # values greater than 2 standard deviations from the mean.
    stddev = math.sqrt(3.0 / (n_inputs + n_outputs))
    return tf.truncated_normal_initializer(stddev=stddev)


learning_rate = 0.1
training_epoches = 25
batch_size = 100
display_step= 1

x = tf.placeholder(tf.float32, [None, 784])     # 28x28 = 784
y = tf.placeholder(tf.float32, [None,  10])

# --------------------------------------------------- #

learning_rate = 0.01
training_epoches = 15

# W1 = tf.Variable(tf.random_normal([784, 256]))
# W2 = tf.Variable(tf.random_normal([256, 256]))
# W3 = tf.Variable(tf.random_normal([256,  10]))
#
# B1 = tf.Variable(tf.random_normal([256]))
# B2 = tf.Variable(tf.random_normal([256]))
# B3 = tf.Variable(tf.random_normal([ 10]))

# W1 = tf.get_variable('W1', shape=[784, 256], initializer=xavier_init(784, 256))
# W2 = tf.get_variable('W2', shape=[256, 256], initializer=xavier_init(256, 256))
# W3 = tf.get_variable('W3', shape=[256,  10], initializer=xavier_init(256,  10))

W1 = tf.get_variable('W1', shape=[784, 256],
                     initializer=tf.contrib.layers.xavier_initializer())
W2 = tf.get_variable('W2', shape=[256, 256],
                     initializer=tf.contrib.layers.xavier_initializer())
W3 = tf.get_variable('W3', shape=[256,  10],
                     initializer=tf.contrib.layers.xavier_initializer())

B1 = tf.Variable(tf.zeros([256]))
B2 = tf.Variable(tf.zeros([256]))
B3 = tf.Variable(tf.zeros([ 10]))

L1 = tf.nn.relu(tf.add(tf.matmul(x, W1), B1))
L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), B2))

activation = tf.add(tf.matmul(L2, W3), B3)
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

            _, c = sess.run([optimizer, cost],
                            feed_dict={x: batch_xs, y: batch_ys})

            # sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            # c = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})

            avg_cost += c / total_batch

        if (epoch+1)%display_step == 0:
            print('epoch: {}, cost: {}'.format(epoch+1, avg_cost))

    # ---------------------------------------- #

    pred = tf.equal(tf.argmax(activation, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(pred, tf.float32))

    print('accuracy :',
          sess.run(accuracy, {x: mnist.test.images, y: mnist.test.labels}))
    # print('accuracy :', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

# 결과
# epoch: 13, cost: 0.1890449569780718
# epoch: 14, cost: 0.18120451631193807
# epoch: 15, cost: 0.17360281172123823
# accuracy : 0.9496

# epoch: 13, cost: 0.18784428118982102
# epoch: 14, cost: 0.18013680148531105
# epoch: 15, cost: 0.17296383041549832
# accuracy : 0.9507
