# -*- coding: utf-8 -*-
# @Time    : 2019/9/27 0027 16:57
# @Author  : 
# @FileName: BP_neural_netWork.py
# @Software: PyCharm

import tensorflow as tf
tf.set_random_seed(111)
# data
x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_data = [[0], [0], [0], [1]]
# X, Y, W, b
X = tf.placeholder(tf.float32, [None, 2], name='X')
Y = tf.placeholder(tf.float32, [None, 1], name='Y')

W1 = tf.Variable(tf.random_normal([2, 2]), dtype=tf.float32, name='weight1')
W2 = tf.Variable(tf.random_normal([2, 1]), dtype=tf.float32, name='weight2')
b1 = tf.Variable(tf.random_normal([2]), name='bias1')
b2 = tf.Variable(tf.random_normal([1]), name='bias2')

# hypothesis
a1 = tf.matmul(X, W1) + b1
a2 = tf.matmul(a1, W2) +b2
# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# print(type(sess.run(b1)))
# sess.close()
# cost
cost = -tf.reduce_mean(Y * tf.log(a2) + (1-Y)*tf.log(1-a2))
# gradientDescent
dz2 = a2 -Y
dw2 = tf.matmul(tf.transpose(a1), dz2)/tf.cast(tf.shape(X)[0], tf.float32)
db2 = tf.reduce_mean(dz2)

dz1 = tf.matmul(dz2, tf.transpose(W2)) * (a1*(1-a1))
dw1 = tf.matmul(tf.transpose(X), dz1)/tf.cast(tf.shape(X)[0], tf.float32)
db1 = tf.reduce_mean(dz1)
# accuracy
prediction = tf.argmax(a2, 1)
correct_prediction = tf.equal(prediction, tf.argmax(y_data, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# update
update = [
    tf.assign(W1, W1-0.001*dw1),
    tf.assign(W2, W2-0.001*dw2),
    tf.assign(b1, b1-0.001*db1),
    tf.assign(b2, b2-0.001*db2)
]
#创建会话
# config=tf.ConfigProto(log_device_placement=True)

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9 # 占用GPU90%的显存
# session = tf.Session(config=config)
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer()) #全局变量初始化
#迭代训练
for step in range(5001):
    cost_val, _, acc = sess.run([cost, update, accuracy], feed_dict={X: x_data, Y: y_data})
    if step % 500 == 0:# 显示损失值收敛情况
        print(step, cost_val, acc)

sess.close()
# 神经网络



"""

# 
# import tensorflow as tf
# import matplotlib.pylab as plt
# tf.set_random_seed(777)
# x_data = [[0.05, 0.1]]
# y_data = [[0.01, 0.99]]
#
# X = tf.placeholder(tf.float32, [None, 2])
# Y = tf.placeholder(tf.float32, [None, 2])
#
# W1 = tf.Variable([[.15, .25],
#                   [.20, .30]])
# b1 = tf.Variable([.35, .35])
# a1 = tf.sigmoid(tf.matmul(X, W1) + b1)
#
# W2 = tf.Variable([[.40, .50],
#                   [.45, .55]])
# b2 = tf.Variable([.60, .60])
# a2 = tf.sigmoid(tf.matmul(a1, W2) + b2)
#
# # cost
# cost = -tf.reduce_mean(Y*tf.log(a2) + (1-Y)*tf.log(1-a2))
# cost_history = []
# dz2 = (a2 - Y)
# dw2 = tf.matmul(tf.transpose(a1), dz2)/tf.cast(tf.shape(a1)[0],
#                                                dtype=tf.float32)
# db2 = tf.reduce_mean(dz2)
#
#
#
# da1 = tf.matmul(dz2, tf.transpose(W2))
# dz1 =  da1* a1 * (1-a1)
# dw1 = tf.matmul(tf.transpose(X), dz1)/tf.cast(tf.shape(X)[0],
#                                               dtype=tf.float32)
# db1 = tf.reduce_mean(dz1, axis=0)
#
# # update
# alpha = 0.1
# update = [
#     tf.assign(W1, W1-alpha*dw1),
#     tf.assign(W2, W2-alpha*dw2),
#     tf.assign(b1, b1-alpha*db1),
#     tf.assign(b2, b2-alpha*db2)
#
# ]
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for i in range(3001):
#         cost_val, _  = sess.run([cost, update],
#                                 feed_dict={X:x_data,
#                                            Y:y_data})
#         if i%100 == 0:
#             print(cost_val)
#         cost_history.append(cost_val)
#     plt.plot(cost_history[1: len(cost_history)])
#     plt.show()
"""

