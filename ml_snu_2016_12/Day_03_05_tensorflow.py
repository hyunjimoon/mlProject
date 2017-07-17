# Day_03_05_tensorflow.py
import tensorflow as tf


def show_constant(c):
    sess = tf.InteractiveSession()
    print(c.eval())
    sess.close()


def show_variable(v):
    sess = tf.InteractiveSession()
    v.initializer.run()
    print(v.eval())
    sess.close()


def show_operation(op):
    sess = tf.InteractiveSession()
    # init = tf.initialize_all_variables()
    init = tf.global_variables_initializer()
    sess.run(init)
    print(sess.run(op))
    sess.close()


a = tf.constant(3)
b = tf.Variable(5)
add = tf.add(a, b)

print(a)
print(b)
print(add)

# # sess = tf.Session()
# sess = tf.InteractiveSession()
# print(a.eval())
#
# b.initializer.run()
# print(b.eval())
#
# sess.close()

show_constant(a)
show_variable(b)
show_operation(add)








