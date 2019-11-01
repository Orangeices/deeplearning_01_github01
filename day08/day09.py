# -*- coding: utf-8 -*-
# @Time    : 2019/10/16 0016 13:34
# @Author  : 
# @FileName: day09.py
# @Software: PyCharm

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import random

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
tf.set_random_seed(111)

# parameter
learning_rate = 0.1
train_epochs = 12
batch_size = 128
total_batch = int(mnist.train.num_examples/batch_size)

print(mnist.test.images[0:1].shape)
# X-->X_img, Y, conv1, conv2, w3
X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28,28,1])
Y = tf.placeholder(tf.float32, [None, 10])

w1 = tf.Variable(tf.random_normal([3, 3, 1, 32]), name='w1')
L1 = tf.nn.conv2d(X_img, w1, strides=[1,1,1,1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

w2 = tf.Variable(tf.random_normal([3, 3, 32, 64]), name='w2')
L2 = tf.nn.conv2d(L1, w2, strides=[1,1,1,1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
L2_flat = tf.reshape(L2, [-1, 7*7*64])

w3 = tf.get_variable('w3', [7*7*64, 10],
                     initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]), name='b')

# hypothesis
logits = tf.matmul(L2_flat, w3) + b

# cost
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# accuracy
predicted = tf.argmax(logits, 1)
correct_pred = tf.equal(predicted, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# save
saver = tf.train.Saver({"Variable":w1, 'Variable_1':w2, "w3":w3, 'Variable_2':b})
# session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # for epoch in range(train_epochs):
    #     avg_cost = 0
    #     for i in range(total_batch):
    #         batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    #         feed_dict = {X: batch_xs, Y: batch_ys}
    #         c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
    #         avg_cost += c / total_batch
    #     print(epoch, avg_cost)
    saver.restore(sess, './ckpt/cnn')


    r = random.randint(0, mnist.test.num_examples-1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
    print("prediction:", sess.run(tf.argmax(logits, 1), feed_dict={X:mnist.test.images[r:r + 1]}))
    # image
    import matplotlib.pyplot as plt
    import numpy as np
    img = plt.imread(r'E:\bawei\DeepLearning_1\深度一\tensorflow补充\1.bmp')
    gravity = np.array([1.,0.,0.])
    greyimg = np.dot(255-img, gravity)
    print("prediction:", sess.run(tf.argmax(logits, 1), feed_dict={X:greyimg.reshape(1, 784)}))
    plt.imshow(greyimg, cmap="Greys", interpolation="nearest")
    plt.show()