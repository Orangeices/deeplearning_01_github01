# -*- coding: utf-8 -*-
# @Time    : 2019/10/25 0025 14:35
# @Author  : 
# @FileName: 12_rnn_mnist.py
# @Software: PyCharm
# import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
# from tensorflow.python.ops.rnn import dynamic_rnn
#
# mnist = input_data.read_data_sets("MNIST_data")
#
#
# batch_size = 100
# n_steps = 28
# n_inputs = 28
# hidden_size = 100
# n_layers = 2
# nb_classes = 10
# learning_rate = 0.01
# n_epochs = 3
#
# # XY
# X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
# Y = tf.placeholder(tf.int32, [None])
# lstm_cells = [tf.contrib.rnn.LSTMCell(num_units=hidden_size) for layer in range(n_layers)]
# multi_cells = tf.contrib.rnn.MultiRNNCell(lstm_cells)
# outputs, last_state = dynamic_rnn(multi_cells, X, dtype=tf.float32)
# logits = tf.contrib.layers.fully_connected(outputs[:, -1], nb_classes, activation_fn=None)
#
# cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=Y))
#
# train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#
# correct = tf.nn.in_top_k(logits, Y, 1)
# accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for epoch in range(n_epochs):
#         for iteration in range(mnist.train.num_examples//batch_size):
#             X_batch, Y_batch = mnist.train.next_batch(batch_size)
#             X_batch = X_batch.reshape((-1, n_steps, n_inputs))
#             sess.run(train, feed_dict={X: X_batch, Y: Y_batch})
#             acc_train = accuracy.eval(feed_dict={X: X_batch, Y: Y_batch})  # 训练准确率
#         acc_test = accuracy.eval(feed_dict={X: mnist.test.images.reshape((-1, n_steps, n_inputs)),
#                                                 Y: mnist.test.labels})  # 测试准确率
#         print(epoch + 1, "Train accuracy:", acc_train, "Test accuracy:", acc_test)

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from tensorflow.python.ops.rnn import dynamic_rnn
mnist = input_data.read_data_sets("MNIST_data")

batch_size = 100
seq_length = 28
nb_classes = 28
hidden_size = 128
learning_rate = 0.1
n_layers = 2

# X
X = tf.placeholder(tf.float32, [None, seq_length, nb_classes])
Y = tf.placeholder(tf.int32, [None])

# cell
cell = [tf.contrib.rnn.LSTMCell(num_units=hidden_size, state_is_tuple=True) for i in range(n_layers)]
multi_cells = tf.contrib.rnn.MultiRNNCell(cell, state_is_tuple=True)
outputs, last_state = dynamic_rnn(multi_cells, X, dtype=tf.float32)

# fully_connected
logits = tf.contrib.layers.fully_connected(outputs[:, -1], hidden_size, None)

cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=Y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.nn.in_top_k(logits, Y, 1)
accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_epochs = 3
    total_batch = int(mnist.train.num_examples//batch_size)

    for epoch in range(train_epochs):
        for i in range(total_batch):
            xs, ys = mnist.train.next_batch(batch_size)
            xs = xs.reshape((-1, 28, 28))
            sess.run(optimizer, feed_dict={X: xs, Y: ys})
            acc_train = accuracy.eval(feed_dict={X: xs, Y: ys})  # 训练准确率
        acc_test = accuracy.eval(feed_dict={X: mnist.test.images.reshape((-1, 28, 28)),
                                                Y: mnist.test.labels})  # 测试准确率
        print(epoch + 1, "Train accuracy: ", acc_train, "Test accuracy: ", acc_test)
