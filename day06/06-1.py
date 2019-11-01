# -*- coding: utf-8 -*-
# @Time    : 2019/10/11 0011 9:43
# @Author  :
# @FileName: 06-1.py
# @Software: PyCharm
# import tensorflow as tf
# tf.set_random_seed(111)
# # data
# x_data = [[1, 2, 1, 1],
#           [2, 1, 3, 2],
#           [3, 1, 3, 4],
#           [4, 1, 5, 5],
#           [1, 7, 5, 5],
#           [1, 2, 5, 6],
#           [1, 6, 6, 6],
#           [1, 7, 7, 7]]
# y_data = [[0, 0, 1],
#           [0, 0, 1],
#           [0, 0, 1],
#           [0, 1, 0],
#           [0, 1, 0],
#           [0, 1, 0],
#           [1, 0, 0],
#           [1, 0, 0]]
# # XYwb
# X = tf.placeholder(tf.float32, [None, 4])
# Y = tf.placeholder(tf.float32, [None, 3])
# w = tf.Variable(tf.random_normal([4, 3]), name='w')
# b = tf.Variable(tf.random_normal([3]), name='b')
#
# # hypothesis
# h = tf.nn.softmax(tf.matmul(X, w) + b)
# # cost
# cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(h), axis=1))
# # train
# train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
# # accuracy
# prediction = tf.argmax(h, axis=1)
# correct_pred = tf.equal(prediction, tf.argmax(y_data, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# # Session
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for i in range(2001):
#         cost_val, _, acc = sess.run([cost, train, accuracy], feed_dict={X:x_data, Y:y_data})
#         if i%200 == 0:
#             print(cost_val, acc)

import tensorflow as tf
import numpy as np

tf.set_random_seed(111)
# data
path = r'E:\bawei\DeepLearning_1\深度一\tensorflow补充'
xy = np.loadtxt(path+'\data-04-zoo.csv', delimiter=',')
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]
# XYwb
nb_classes = 7
X = tf.placeholder(tf.float32, [None, 16])
Y = tf.placeholder(tf.int32, [None, 1])
Y_one_hot = tf.one_hot(Y, nb_classes)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
w = tf.Variable(tf.random_normal([16, nb_classes]), name='w')
b = tf.Variable(tf.random_normal([nb_classes]), name='b')
# hypothesis
logits = tf.matmul(X, w) + b
hypothesis = logits
# cost
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y_one_hot))
# train
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
# accuracy
prediction = tf.argmax(hypothesis, 1)
correct_pred = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2001):
        cost_val, _, acc = sess.run([cost, train, accuracy], feed_dict={X:x_data,Y:y_data})
        if i%100 == 0:
            print(cost_val, acc)

