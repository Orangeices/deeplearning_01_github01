# -*- coding: utf-8 -*-
# @Time    : 2019/10/13 0013 13:49
# @Author  : 
# @FileName: 07_practice01.py
# @Software: PyCharm
# #请利用Tensorflow实现Softmax多分类。
# import tensorflow as tf
#
# # 2. 题目要求：
# # ①　导入必要的依赖库，同时设置随机种子。（5分）
# tf.set_random_seed(111)
# # ②　定义数据集，其中：
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
#
# # ③　定义占位符。（5分）
# X = tf.placeholder(tf.float32, [None, 4])
# Y = tf.placeholder(tf.float32, [None, 3])
# # ④　定义权重W和偏置b。（5分）
# w = tf.Variable(tf.random_normal([4, 3]), name='w')
# b = tf.Variable(tf.random_normal([3]), name='b')
# # ⑤　定义预测模型，激活函数采用softmax。（5分）
# hypothesis = tf.nn.softmax(tf.matmul(X, w) + b)
# # ⑥　定义代价或损失函数。（5分）
# cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), 1))
#
# # ⑦　定义梯度下降优化器，学习率设置为0.01。（5分）
# train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
#
# # ⑧　准确率计算。（5分）
# predicted = tf.argmax(hypothesis, 1)
# correct_pred = tf.equal(predicted, tf.argmax(Y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# # ⑨　创建会话，并全局变量初始化。（5分）
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for i in range(5001):
#         cost_val, _, acc = sess.run([cost, train, accuracy], feed_dict={X: x_data,Y: y_data})
#         if i%500 == 0:
#             print(cost_val)
# # ⑩　开始迭代训练，循环迭代5000次，每500次打印输出一次损失值的收敛情况。（5分）
# # 11　打印输出模型的准确率。（5分）
#     print('输出模型的准确率:',acc )
# # 12　给定一组x:[[4,1,2,3], [3,2,4,5]]，对该组数据进行测试，并打印输出测试结果。（5分）
#     p1, h1 = sess.run([predicted, hypothesis], feed_dict={X:[[4,1,2,3], [3,2,4,5]]})
#     print(p1, '\n', h1)

# ①　导入所需库文件。（5分）


""""""
# import tensorflow as tf
# import numpy as np
# tf.set_random_seed(111)
# np.random.seed(111)
# # ②　调用np.array生成广告花费（money）和点击量数据（click）；其中
# # money = np.array([[109],[82],[99], [72], [87], [78], [86], [84], [94], [57]]；click：[[11], [8], [8], [6],[ 7], [7], [7], [8], [9], [5]]。（5分）
# money = np.array([[109],[82],[99], [72], [87], [78], [86], [84], [94], [57]])
# click = np.array([[11], [8], [8], [6],[ 7], [7], [7], [8], [9], [5]])
# # ③　划分训练集和测试集数据。（5分）
# order = np.random.permutation(10)
# x = money[order]
# y = click[order]
# d = int(money.shape[0] * 0.7)
# x_train, x_test = np.split(x, [d, ])
# y_train, y_test = np.split(y, [d, ])
#
# # ④　设置自变量x的占位符和因变量y_的占位符；同时设置权重值W变量和偏置量b变量。（5分）
# x = tf.placeholder(tf.float32, [None, 1])
# y = tf.placeholder(tf.float32, [None, 1])
# w = tf.Variable(tf.random_normal([1, 1]), name='w')
# b = tf.Variable(tf.random_normal([1]), name='b')
# # ⑤　设置线性模型：y=Wx+b。（5分）
# hypothesis = tf.matmul(x, w) + b
# # ⑥　设置成本函数(最小方差)：
# # （提示：成本函数为tf.reduce_sum(tf.pow((y_-y),2)))；
# # 同时使用梯度下降，以0.000001的学习速率最小化成本函数cost，以获得W和b的值。（5分）
# cost = tf.reduce_mean(tf.square(hypothesis - y))
# train = tf.train.AdamOptimizer(learning_rate=0.7).minimize(cost)
# # ⑦　创建会话，初始化全局变量。（5分）
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for i in range(101):
#         cost_val, _, w_, b_ = sess.run([cost, train, w, b], feed_dict={x:x_train, y:y_train})
#         print(cost_val, w_, b_)
# # ⑧　循环训练模型100次，打印输出每次训练后的W，b和cost值和最终的W，b和cost值。（5分）
#     h1 = sess.run(hypothesis, feed_dict={x:x_test})
#     print(h1, '\n', y_test)

import numpy as np
import tensorflow as tf
tf.set_random_seed(111)
path = r'E:\bawei\DeepLearning_1\深度一\tensorflow补充'
xy = np.loadtxt(path+'\data-04-zoo.csv', delimiter=',')
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]
# XYwb
X = tf.placeholder(tf.float32, [None, 16])
Y = tf.placeholder(tf.int32, [None, 1])
Y_one_hot = tf.one_hot(Y, 7)
Y_one_hot = tf.reshape(Y_one_hot, [-1, 7])
w = tf.Variable(tf.random_normal([16, 7]), name='w')
b = tf.Variable(tf.random_normal([7]), name='b')
# hypothesis \
logits = tf.matmul(X, w) + b
hypothesis = logits
# cost
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=Y_one_hot))
# train
train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
# accuracy
predicted = tf.argmax(hypothesis, 1)
correct_pred = tf.equal(predicted, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# Session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2001):
        cost_val, _, acc = sess.run([cost, train, accuracy], feed_dict={X:x_data, Y:y_data})
        if i%100 == 0:
            print(cost_val, acc)
