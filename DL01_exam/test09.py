# -*- coding: utf-8 -*-
# @Time    : 2019/10/15 0015 8:37
# @Author  : 
# @FileName: test09.py
# @Software: PyCharm

# 在tensorflow下，训练一个softmax分类器，实现数据集的多分类功能
# （一）导入tensorflow模块，设置随机种子（8分）
import tensorflow as tf
import numpy as np
tf.set_random_seed(111)
# （二）准备训练数据集x_data、y_data，从文件data-04-zoo.csv中读取(8分)
xy = np.loadtxt(r'data-04-zoo.csv', delimiter=',')
# print(xy)
x_data , y_data = xy[:, 0:-1], xy[:, [-1]]
nb_classes = 7
# （三）定义张量X和Y，float32类型，使用占位符函数（8分）
# print(x_data.shape)
X = tf.placeholder(tf.float32, [None, 16])
Y = tf.placeholder(tf.int32, [None, 1])
# （四）把Y转换为独热编码（8分）
Y_one_hot = tf.one_hot(Y, nb_classes)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])

# （五）定义张量W（weight）和b（bias）（8分）
w = tf.Variable(tf.random_normal([16, nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes]))
# （六）定义hypothesis预测模型（8分）
logits = tf.matmul(X, w) + b
hypothesis = logits
# （七）定义代价函数（损失函数）（8分）
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y_one_hot))
# （八）使用梯度下降优化器计算最小费用，查找最优解（8分）
train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(cost)
# （九）创建会话（Session），全局变量初始化（8分）
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
# （十）开始迭代总共2001次（6分）
    for i in range(2001):
        cost_val, _ = sess.run([cost, train], feed_dict={X:x_data, Y:y_data})
# （十一）使用训练集的数据进行训练（8分）
# （十二）每100次输出一次cost值（6分）
        if i%100 == 0:
            print(cost_val)
# （十三）使用最后一个训练集的样本做为测试样本，进行分类测试，输出分类结果（8分）
    p1 = sess.run(tf.argmax(logits, 1), feed_dict={X:x_data[-2:-1]})
    print(p1)
