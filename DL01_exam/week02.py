# -*- coding: utf-8 -*-
# @Time    : 2019/10/14 0014 8:37
# @Author  : 
# @FileName: week02.py
# @Software: PyCharm
# 请使用tensorflow，实现如下要求。
# 2. 要求：
# (1)	导入必要的依赖库。（8分）
import tensorflow as tf
# (2)	设置随机种子（8分）
tf.set_random_seed(111)
# (3)	定义权重与偏执变量W,b（8分）
w = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
# (4)	定义输入值x:[1,2,3] （8分）
x_data = [1,2,3]
# (5)	定义目标值y:[1,2,3] （8分）
y_data = [1,2,3]
# (6)	定义输入占位符X。（8分）
X = tf.placeholder(tf.float32, [None])
# (7)	定义输出占位符Y。（8分）
Y = tf.placeholder(tf.float32, [None])
# (8)	定义线性激活函数进行预测。（8分）
hypothesis = tf.nn.relu(tf.multiply(X, w) + b)
# (9)	定义代价损失函数（8分）
cost = tf.reduce_mean(tf.square(hypothesis - Y))
# (10)	模型训练，利用优化函数进行优化（8分）
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
# (11)	创建会话（8分）
with tf.Session() as sess:
    # (12)	全局变量初始化（8分）
    sess.run(tf.global_variables_initializer())
# (13)	迭代训练2000次，每20次输出一次loss值（4分）
    for i in range(2001):
        cost_val, _ = sess.run([cost, train], feed_dict={X:x_data, Y:y_data})
        if i%20 == 0:
            print('step:', i, 'loss:', cost_val)
    # w_, b_ = sess.run([w, b])
    print(w.eval(), b.eval())