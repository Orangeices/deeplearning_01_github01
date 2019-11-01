# -*- coding: utf-8 -*-
# @Time    : 2019/10/11 0011 8:36
# @Author  : 
# @FileName: test06.py
# @Software: PyCharm
# （一）导入tensorflow模块，设置随机种子（8分）
import tensorflow as tf
tf.set_random_seed(111)
# （二）准备数据集（8分）
x_data = [[0, 0],
          [0, 1],
          [1, 0],
          [1, 1]]
y_data = [[0],
          [1],
          [1],
          [0]]
# x1,x2,y的数据是0或者1，如下表，x1和x2进行逻辑异或的结果是y
# x1	x2	Y
# 0	0	0
# 0	1	1
# 1	0	1
# 1	1	0
# （三）初始化X，Y占位符（8分）
X = tf.placeholder(tf.float32, [None, 2])
Y = tf.placeholder(tf.float32, [None, 1])
# （四）初始化W1，b1张量（8分）
w1 = tf.Variable(tf.random_normal([2, 2]))
b1 = tf.Variable(tf.random_normal([2]))
# （五）设置隐藏层layer1模型，使用sigmoid函数（8分）
# （六）初始化W2，b2张量（8分）
w2 = tf.Variable(tf.random_normal([2, 1]))
b2 = tf.Variable(tf.random_normal([1]))
# （七）设置hypothesis预测模型（8分）

a1 = tf.sigmoid(tf.matmul(X, w1) + b1)
a2 = tf.sigmoid(tf.matmul(a1, w2) + b2)
# （八）设置代价函数（8分）
cost = -tf.reduce_mean(Y*tf.log(a2) + (1-Y)*tf.log(1-a2))
# （九）不能使用梯度下降优化器，自己编写底层代码实现BP反向传播
dz2 = a2 - Y
dw2 = tf.matmul(tf.transpose(a1), dz2) / tf.cast(tf.shape(a1)[0], tf.float32)
db2 = tf.reduce_mean(dz2)

dz1 = tf.matmul(dz2, tf.transpose(w2)) * (a1*(1-a1))
dw1 = tf.matmul(tf.transpose(X), dz1) / tf.cast(tf.shape(X)[0], tf.float32)
db1 = tf.reduce_mean(dz1)
learning_rate = 0.1
update = [
    tf.assign(w2, w2-learning_rate*dw2),
    tf.assign(b2, b2-learning_rate*db2),
    tf.assign(w1, w1-learning_rate*dw1),
    tf.assign(b1, b1-learning_rate*db1)
]
#
predicted = tf.cast(a2>0.5, tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), tf.float32))
# 7个公式对应7行代码，每行代码3分，（共21分）
# （十）创建会话，初始化全局变量（5分）
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2001):
        _, cost_val, acc = sess.run([update, cost, accuracy], feed_dict={X:x_data, Y:y_data})
        if i%200 == 0:
            print(cost_val, acc)
# （十一）迭代训练2000次，每200次输出一次cost（5分）
# （十二）输出预测值、准确度（5分）

