# -*- coding: utf-8 -*-
# @Time    : 2019/9/26 0026 13:03
# @Author  : 
# @FileName: demo01.py
# @Software: PyCharm
import tensorflow as tf
import numpy as np

tf.set_random_seed(111)
# 建立字典
idx2char = ['h','i','e','l','o']
# 构造数据集
x_data = [0,1,1,2,3,3]
x_one_hot = np.eye(5)[x_data].reshape(1, -1, 5)  # (1, 6, 5)
y_data = [[1,0,2,3,3,4]]


# parameter
batch_size = 1
sequence_length = 6
input_dim = 5
hidden_size = 8
num_classes = 5
learning_rate = 0.1

X = tf.placeholder(tf.float32, [None, sequence_length, input_dim])  # (？， 6， 5）
Y = tf.placeholder(tf.int32, [None, sequence_length])  # （？， 6）


cell = tf.contrib.rnn.LSTMCell(num_units=hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, X, initial_state=initial_state, dtype=tf.float32)
# print(outputs)
# print(_states)
X_for_fc = tf.reshape(outputs, [-1, hidden_size])
outputs = tf.contrib.layers.fully_connected(inputs=X_for_fc,
                                            num_outputs=num_classes, activation_fn=None)

# print(outputs)
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])

weights = tf.ones([batch_size, sequence_length])

sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
# sess = tf.Session()
# print(sess.run(outputs, feed_dict={X:x_one_hot,Y:y_data}))



# sess.close()