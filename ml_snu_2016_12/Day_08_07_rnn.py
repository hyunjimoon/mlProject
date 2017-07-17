# Day_08_07_rnn.py
import tensorflow as tf
import numpy as np

# helo
unique = ['h', 'e', 'l', 'o']   #'helo'
y_data = [1, 2, 2, 3]   # e, l, l, o
x_data = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,1,0]], # hell
                  dtype=np.float32)
print(x_data)

cells  = tf.nn.rnn_cell.BasicRNNCell(4)
state  = tf.zeros([1, cells.state_size])    # (1,4)
x_data = tf.split(0, 4, x_data)

print(x_data)
# [<tf.Tensor 'split:0' shape=(1, 4) dtype=float32>,
#  <tf.Tensor 'split:1' shape=(1, 4) dtype=float32>,
#  <tf.Tensor 'split:2' shape=(1, 4) dtype=float32>,
#  <tf.Tensor 'split:3' shape=(1, 4) dtype=float32>]

outputs, state = tf.nn.rnn(cells, x_data, state)
print(outputs)      # [(1, 4), (1, 4), (1, 4), (1, 4)]
# [<tf.Tensor 'RNN/BasicRNNCell/Tanh:0' shape=(1, 4) dtype=float32>,
#  <tf.Tensor 'RNN/BasicRNNCell_1/Tanh:0' shape=(1, 4) dtype=float32>,
#  <tf.Tensor 'RNN/BasicRNNCell_2/Tanh:0' shape=(1, 4) dtype=float32>,
#  <tf.Tensor 'RNN/BasicRNNCell_3/Tanh:0' shape=(1, 4) dtype=float32>]

print(state)        # (1, 4)

logits  = tf.reshape(tf.concat(1, outputs), [-1, 4])    # (4, 4)
targets = tf.reshape(y_data, [-1])
weights = tf.ones([4])
print(logits)

loss = tf.nn.seq2seq.sequence_loss_by_example([logits], [targets], [weights])
cost = tf.reduce_sum(loss)
train = tf.train.RMSPropOptimizer(0.01, 0.9).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(100):
        sess.run(train)
        r0, r1, r2, r3 = sess.run(tf.argmax(logits, 1))
        print(r0, r1, r2, r3, ':', unique[r0], unique[r1], unique[r2], unique[r3])

# https://tensorflowkorea.gitbooks.io/tensorflow-kr/content/g3doc/tutorials/recurrent



