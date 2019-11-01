# -*- coding: utf-8 -*-
# @Time    : 2019/10/10 0010 8:36
# @Author  : 
# @FileName: test05.py
# @Software: PyCharm

# 使用tensorflow框架，建立神经网络，实现逻辑或的功能。
import tensorflow as tf
# （一）导入tensorflow模块，设置随机种子（8分）
tf.set_random_seed(111)
# （二）准备数据集（8分）
x_data = [[0, 0],
          [0, 1],
          [1, 0],
          [1, 1]]
y_data = [[0],
          [1],
          [1],
          [1]]

#
#
# x1,x2,y的数据是0或者1，如下表，x1和x2进行逻辑或的结果是y
# x1	x2	y
#
# 0	0	0
# 0	1	1
# 1	0	1
# 1	1	1
# （三）初始化X，Y占位符（8分）
X = tf.placeholder(tf.float32, [None, 2])
Y = tf.placeholder(tf.float32, [None, 1])
# （四）初始化W1，b1张量（8分）
w1 = tf.Variable(tf.random_normal([2, 2]), name='w1')
b1 = tf.Variable(tf.random_normal([2]), name='b1')
# （五）设置隐藏层模型，使用sigmoid函数（8分）
# （六）初始化W2，b2张量（8分）
w2 = tf.Variable(tf.random_normal([2, 1]), name='w2')
b2 = tf.Variable(tf.random_normal([1]), name='b2')
# （七）设置hypothesis预测模型（8分）
a1 = tf.sigmoid(tf.matmul(X, w1) + b1)
a2 = tf.sigmoid(tf.matmul(a1, w2) + b2)
# （八）设置代价函数（8分）
cost = -tf.reduce_mean(Y*tf.log(a2) + (1-Y)*tf.log(1-a2))
# accuracy
predicted = tf.cast(a2>0.5, tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y_data), tf.float32))

# （九）使用梯度下降优化器查找最优解（8分）
train = tf.train.GradientDescentOptimizer(learning_rate=0.02).minimize(cost)
# （十）创建会话（6分）
with tf.Session() as sess:
# （十一）初始化全 局变量（6分）
    sess.run(tf.global_variables_initializer())
# （十二）迭代训练1000次，每100次输出一次cost（8分）
    for i in range(1001):
        cost_val, _, acc, pred = sess.run([cost, train, accuracy, predicted], feed_dict={X:x_data, Y:y_data})
        if i%100 == 0:
            print(cost_val, acc, pred)
# 输出预测值、准确度（8分）