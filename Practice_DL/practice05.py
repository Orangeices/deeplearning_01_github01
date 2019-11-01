# -*- coding: utf-8 -*-
# @Time    : 2019/10/20 0020 20:21
# @Author  : 
# @FileName: practice05.py
# @Software: PyCharm
# import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
# import random
# import matplotlib.pyplot as plt
# import numpy as np
#
# tf.set_random_seed(111)
# # data
# mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
# # parameter
# learning_rate = 0.001
# training_epochs = 15
# batch_size = 100
# # XYwb
# X = tf.placeholder(tf.float32, [None, 784])
# X_img = tf.reshape(X, [-1, 28,28,1])
# Y = tf.placeholder(tf.float32, [None, 10])
#
# # layer1
# w1 = tf.Variable(tf.random_normal([3,3,1,32]), name='w1') # filter
# L1 = tf.nn.conv2d(X_img, w1, strides=[1,1,1,1], padding='SAME')
# L1 = tf.nn.relu(L1)
# L1 = tf.nn.max_pool(L1,[1,2,2,1], strides=[1,2,2,1], padding='SAME')
# # layer2
# w2 = tf.Variable(tf.random_normal([3,3,32,64]), name='w2')
# L2 = tf.nn.conv2d(L1, w2, strides=[1,1,1,1], padding='SAME')
# L2 = tf.nn.relu(L2)
# L2 = tf.nn.max_pool(L2, [1,2,2,1],strides=[1,2,2,1], padding='SAME')
#
# # full_connect
# dim = L2.get_shape()[1].value*L2.get_shape()[2].value*L2.get_shape()[3].value
# L2_flat = tf.reshape(L2, [-1, dim])
# w3 = tf.get_variable('w3', [dim, 10], initializer=tf.contrib.layers.xavizer_initializer())
# b = tf.Variable(tf.random_normal([10]), name='b')
# logits = tf.matmul(L2_flat, w3)+b
# # cost
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
# # optimizer
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#
# tf.summary.scalar('loss', cost)
# summary = tf.summary.merge_all()
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     writer = tf.summary.FileWriter('./logs/tensorboard1', sess.graph)
#     for epoch in range(training_epochs):
#         avg_cost = 0
#         total_batch = int(mnist.train.num_examples/batch_size)
#         for i in range(total_batch):
#             batch_xs, batch_ys = mnist.train.next_batch()
#             c, _, s = sess.run([cost, optimizer, summary], feed_dict={X:batch_xs,Y:batch_ys})
#             avg_cost += c/total_batch
#             writer.add_summary(s, global_step=i)
#         print("")

import numpy as np

X=np.array([[0,0,1],
            [1,0,1],
            [1,1,1],
            [0,1,1]])
Y=np.array([[0,1,1,0]]).T
def nonlin(x, diver=False):
    if diver:
        return x * (1-x)
    else:
        return 1/(1+np.exp(-x))

def costF(x, y, w, b):
    m, l = x.shape
    a = nonlin(x.dot(w) + b)
    J = (-1/m) * (y.T.dot(np.log(a)) + (1-y).T.dot(np.log(1-a)))
    return J

def gd(x, y, n=3, alpha=0.1, iterN=1000):
    m, l = x.shape
    w1 = 2*np.random.random((l, n))-1
    w2 = 2*np.random.random((n, 1))-1
    b1 = 0.01
    b2 = 0.02
    jArr = np.zeros(iterN)
    for i in range(iterN):
        a1 = nonlin(x.dot(w1) + b1)
        a2 = nonlin(a1.dot(w2) + b2)
        jArr[i] = costF(a1, y, w2, b2)
        dz2 = a2 - y
        dw2 = (1/m)*(a1.T.dot(dz2))
        db2 = np.mean(dz2)

        dz1 = dz2.dot(w2.T) * nonlin(a1, True)
        dw1 = (1/m)*(x.T.dot(dz1))
        db1 = np.mean(dz1)

        w2 -= alpha*dw2
        w1 -= alpha*dw1
        b2 -= alpha*db2
        b1 -= alpha*db1

    return w1,w2,b1,b2, jArr

w1,w2,b1,b2, jarr = gd(X, Y)
import matplotlib.pyplot as plt
plt.plot(jarr)
plt.show()