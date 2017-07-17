# Day_09_03_ImageProcessing_1.py
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def decode_image(image_filenames, resize_func=None):
    name = tf.placeholder(dtype=tf.string)
    file = tf.read_file(name)
    image = tf.image.decode_jpeg(file)

    if resize_func:
        image = resize_func(image)

    images = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i, filename in enumerate(image_filenames, 1):
            images.append(sess.run(image, feed_dict={name: filename}))

            if i%1000 == 0:
                print(i, 'processed.')

        return images

TRAIN_DIR = 'CatDog/train/'
TEST_DIR  = 'CatDog/test/'

# 컴프리헨션 comprehension
# for i in os.listdir(TRAIN_DIR)
#     TRAIN_DIR + i

train_files = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)]
test_files  = [ TEST_DIR+i for i in os.listdir( TEST_DIR)]

if '.DS_Store' in train_files[0]:
    train_files = train_files[1:]

# one-hot encoding
labels = [[1.,0.] if 'dog' in name else [0.,1.] for name in train_files]

train_images = decode_image(train_files)
test_images  = decode_image(test_files)
all_images   = train_images + test_images

width, height, ratio = [], [], []
for image in all_images:
    h, w, _ = image.shape
    ratio.append(w/h)
    width.append(w)
    height.append(h)

# plt.plot(ratio)
# plt.show()

plt.plot(width, height, '.r')
plt.show()







