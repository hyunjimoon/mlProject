# Day_03_06_tensorflow_2.py
import tensorflow as tf

def not_used():
    value  = tf.Variable(0)
    one    = tf.constant(1)
    state  = tf.add(value, one)
    update = tf.assign(value, state)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(3):
        # print(sess.run(update), sess.run(value), sess.run(state))
        print(sess.run(state), sess.run(update), sess.run(value))
        # print(sess.run(state))

    sess.close()


def not_used_2():
    a = tf.constant(3)
    b = tf.constant(5)
    # add = tf.add(a, b)

    x = tf.placeholder(tf.int32)
    y = tf.placeholder(tf.int32)

    add = tf.add(x, y)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print(sess.run(add, feed_dict={x: 3, y: 5}))
    feed = {x: [3,5], y: [2,7]}
    # feed = dict(x=5, y=7)
    print(sess.run(add, feed_dict=feed))
    sess.close()


# 문제
# 구구단의 특정 단을 출력하는 함수를 만드세요.
def NineNine(dan):
    # first  = tf.placeholder(tf.int32)
    # second = tf.placeholder(tf.int32)
    # mul = first * second # tf.mul(first, second)
    #
    # sess = tf.Session()
    # sess.run(tf.global_variables_initializer())
    #
    # for i in range(1, 10):
    #     # print('{} x {} = {}'.format(i, dan, i*dan))
    #     print('{} x {} = {}'.format(i, dan,
    #                                 sess.run(mul, feed_dict={first: i, second: dan})))
    # sess.close()

    first  = tf.placeholder(tf.int32)
    second = tf.constant(dan)
    mul = first * second # tf.mul(first, second)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1, 10):
        # print('{} x {} = {}'.format(i, dan, i*dan))
        print('{} x {} = {}'.format(dan, i,
                                    sess.run(mul, feed_dict={first: i})))
    sess.close()

# NineNine(7)
not_used()

# 과제
# add만으로 구현하기

















