# -*- coding: utf-8 -*-
# @Time    : 2019/10/28 0028 8:36
# @Author  : 
# @FileName: test20.py
# @Software: PyCharm
# 在tensorflow中，用循环神经网络实现mnist手写数字识别。
# import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
# from tensorflow.python.ops.rnn import dynamic_rnn
# # 1.	读取数据（8分）
# mnist = input_data.read_data_sets("MNIST_data")
# # 2.	设置参数（8分）
# learning_rate = 0.01
# batch_size = 100
# epochs = 15
# hidden_size = 100
# # 3.	设置占位符（8分）
# X = tf.placeholder(tf.float32, [None, 28, 28])
# Y = tf.placeholder(tf.int32, [None])
# # 4.	建立LSTMCell（8分）
# cells = [tf.contrib.rnn.LSTMCell(num_units=hidden_size, state_is_tuple=True) for _ in range(2)]
# # 5.	堆叠多层LSTMCell（8分）
# multi_cells = tf.contrib.rnn.MultiRNNCell(cells,state_is_tuple=True)
#
# outputs, last_size = dynamic_rnn(multi_cells, X, dtype=tf.float32)
# # 6.	建立全连接层（8分）
# outputs = tf.contrib.layers.fully_connected(outputs[:, -1], 10, activation_fn=None)
# # 7.	计算代价或损失函数（8分）
# logits = outputs
# cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=Y))
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# # 8.	设置准确率模型（8分）
# predicted = tf.nn.in_top_k(logits, Y, 1)
# accuracy = tf.reduce_mean(tf.cast(predicted, tf.float32))
# total_batch = int(mnist.train.num_examples/batch_size)
# # 9.	用训练集数据一共训练迭代15次（8分）
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for epoch in range(epochs):
#         acc_train = 0
#         for i in range(total_batch):
#             xs, ys = mnist.train.next_batch(batch_size)
#             xs = xs.reshape((-1, 28, 28))
#             c, _, acc = sess.run([cost, optimizer, accuracy],
#                                  feed_dict={X:xs, Y:ys})
# # 10.	分批次训练，每批100个训练样本（8分）
#             acc_train += c/total_batch
# # 11.	用训练集计算准确率（6分）
#         print(acc)
# # 12.	用测试集计算准确率（6分）
#     print("ac_test:  ", sess.run(accuracy,
#                                  feed_dict={X:mnist.test.images[0:500].reshape((-1, 28, 28)), Y:mnist.test.labels[0:500]}))
# # 13.	从测试集里抽一个样本，用循环神经网络进行验证（8分）
#     import random
#     r = random.randint(0, mnist.test.num_examples-1)
#     print(sess.run(outputs, feed_dict={X:mnist.test.images[r:r+1].reshape((-1, 28, 28)),
#                                            Y:mnist.test.labels[r:r+1]}))


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.datasets import load_iris
data = load_iris()
x = data.data
stand = MinMaxScaler()
x = stand.fit_transform(x)
y = data.target

def nonlin(x, deriv=False):
    if deriv:
        return x*(1-x)
    else:
        return 1/(1+np.exp(-x))

def costF(x,y,w,b):
    m,l = x.shape
    a = nonlin(x.dot(w) + b)
    J = (-1/m)*(y.T.dot(np.log(a)) + (1-y).T.dot(np.log(1-a)))
    return J


def gd(x,y,n=20, iterN=7000, alpha=0.0001):
    m, l = x.shape
    w1 = 2*np.random.random((l, n))-1
    w2 = 2*np.random.random((n, 1))-1
    b1 = 0
    b2 = 0

    jArr = np.zeros(iterN)
    for i in range(iterN):
        a1 = nonlin(x.dot(w1)+b1)
        a2 = nonlin(a1.dot(w2)+b1)

        jArr[i] = costF(a1, y, w2, b2)

        dz2 = a2 - y
        dw2 = (1/m) * (a1.T.dot(dz2))
        db2 = np.mean(dz2)

        dz1 = dz2.dot(w2.T) * (nonlin(a1, True))
        dw1 = (1/m)*(x.T.dot(dz1))
        db1 = np.mean(dz1)

        w2 -= alpha * dw2
        w1 -= alpha * dw1
        b2 -= alpha * db2
        b1 -= alpha * db1
    return w1, w2, b1, b2, jArr

w1, w2, b1, b2, jarr = gd(x,y.reshape(150, 1))
import matplotlib.pyplot as plt
plt.plot(jarr)
plt.show()