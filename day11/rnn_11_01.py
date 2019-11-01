# -*- coding: utf-8 -*-
# @Time    : 2019/10/21 0021 13:43
# @Author  : 
# @FileName: rnn_11_01.py
# @Software: PyCharm
import numpy as np
"""


x = [1,2]
state = [0.0, 0.0]
w_cell_state = np.asarray([[0.1, 0.2],
                           [0.3, 0.4]])
w_cell_input = np.asarray([0.5, 0.6])
b_cell = np.asarray([0.1, -0.1])
w_output = np.asarray([1.0, 2.0])
b_output = 0.1
for i in range(len(x)):
    before_activation = np.dot(state, w_cell_state) + x[i] * w_cell_input + b_cell
    state = np.tanh(before_activation)
    final_output = np.dot(state, w_output) + b_output
    print("before activation:", before_activation)
    print("state:", state)
    print("output:", final_output)



# before activation: [0.6 0.5]
# state: [0.53704957 0.46211716]
# output: 1.561283881518055
# before activation: [1.2923401  1.39225678]
# state: [0.85973818 0.88366641]
# output: 2.727071008233731

"""


import tensorflow as tf
# 手动实现RNN：rnn_01.py
tf.set_random_seed(777)
#
# # data
# x0_batch = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 0, 1]])
# x1_batch = np.array([[9, 8, 7], [0, 0, 0], [6, 5, 4], [3, 2, 1]])
# # parameter
# n_inputs = 3
# n_neurons = 5
# # x1,x2
# x0 = tf.placeholder(tf.float32, [None, n_inputs])# (4, 3)
# x1 = tf.placeholder(tf.float32, [None, n_inputs])
#
# wx = tf.Variable(tf.random_normal([n_inputs, n_neurons],dtype=tf.float32)) # (3,5)
# wy = tf.Variable(tf.random_normal([n_neurons, n_neurons], dtype=tf.float32))
# b = tf.Variable(tf.random_normal([1,n_neurons], dtype=tf.float32))
# y0 = tf.tanh(tf.matmul(x0, wx) + b)
# y1 = tf.tanh( tf.matmul(y0, wy)+tf.matmul(x1, wx) + b )
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     y0_val, y1_val = sess.run([y0, y1], feed_dict={x0:x0_batch, x1:x1_batch})
#     print("Y0_val:\n", y0_val)
#     print("Y1_val:\n", y1_val)


"""
n_inputs = 3
n_neurons = 5
x0 = tf.placeholder(tf.float32, [None, n_inputs])
x1 = tf.placeholder(tf.float32, [None, n_inputs])

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
output_seqs, states = tf.contrib.rnn.static_rnn(basic_cell, [x0, x1], dtype=tf.float32)
y0, y1 = output_seqs
x0_batch = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 0, 1]])  #(4,3)
x1_batch = np.array([[9, 8, 7], [0, 0, 0], [6, 5, 4], [3, 2, 1]])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    Y0_val, Y1_val, os, sta = sess.run([y0, y1, output_seqs, states], feed_dict={x0: x0_batch, x1: x1_batch})

# print("Y0_val:\n",Y0_val)
print("Y1_val:\n",Y1_val, '\n')
print(sta)


"""

n_steps = 2
n_inputs = 3
n_neurons = 5
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)
X_batch = np.array([
    # t = 0　　 t = 1
    [[0, 1, 2], [9, 8, 7]], # instance 1
    [[3, 4, 5], [0, 0, 0]], # instance 2
    [[6, 7, 8], [6, 5, 4]], # instance 3
    [[9, 0, 1], [3, 2, 1]], # instance 4
])
# print(X_batch.shape)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    outputs_val, states_val = sess.run([outputs, states], feed_dict={X: X_batch})
    print('outputs_val')
    print(outputs_val) #(4,2,5)
    print('states_val')
    print(states_val) #(4,5)
    print('outputs_val[:,-1]')
    print(outputs_val[:,-1]) #(4,5)
