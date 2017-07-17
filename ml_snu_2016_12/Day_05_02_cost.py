# Day_05_02_cost.py
import tensorflow as tf
import matplotlib.pyplot as plt

# 문제
# Day_02_03_cost.py 함수의 코드를 텐서플로우 버전으로 변환해 보세요.

x = [1, 2, 3]
y = [1, 2, 3]

W = tf.placeholder(tf.float32)
hypo = W * x
cost = tf.reduce_mean(tf.square(hypo-y))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

xx, yy = [], []
for i in range(-30, 50):
    c = sess.run(cost, feed_dict={W: i/10})

    print(i/10, c)

    xx.append(i/10)
    yy.append(c)

# plt.plot(x, y)
plt.plot(xx, yy, 'ro')
plt.show()

sess.close()

