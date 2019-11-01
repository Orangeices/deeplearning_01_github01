# -*- coding: utf-8 -*-
# @Time    : 2019/10/11 0011 10:31
# @Author  : 
# @FileName: 06-2.py
# @Software: PyCharm
# MNIST手写数字识别的卷积神经网络版本
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

tf.set_random_seed(777) #设置随机种子

# 获取数据集
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# 参数
learning_rate = 0.001 # 学习率
training_epochs = 15  # 训练总周期
batch_size = 100 # 训练每批样本数

#定义占位符
X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1])      # 28 x 28 灰度图片
Y = tf.placeholder(tf.float32, [None, 10])  # 独热编码

# 第1层卷积，输入图片数据(?, 28, 28, 1)
# W1 = tf.Variable(tf.random_normal([3, 3, 1, 32]))  #卷积核3x3，输入通道1，输出通道32
# L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME') #卷积输出 （?, 28, 28, 32)
# L1 = tf.nn.relu(L1)
# L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME') #池化输出 (?, 14, 14, 32)

# 第2层卷积，输入图片数据(?, 14, 14, 32)
W2 = tf.Variable(tf.random_normal([3, 3, 1, 64], stddev=0.01)) #卷积核3x3，输入通道32，输出通道64
L2 = tf.nn.conv2d(X_img, W2, strides=[1, 1, 1, 1], padding='SAME') #卷积输出  (?, 14, 14, 64)
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #池化输出 (?, 7, 7, 64)

L2_flat = tf.reshape(L2, [-1, 14 * 14 * 64])  # 变成一维向量 (?, 3136)

# 全连接 7x7x64 inputs -> 10 outputs
W3 = tf.get_variable("W3", shape=[14 * 14 * 64, 10], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]))
logits = tf.matmul(L2_flat, W3) + b

#代价或损失函数
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # 优化器

# 创建会话
sess = tf.Session()
sess.run(tf.global_variables_initializer()) #全局变量初始化
# 迭代训练
print('开始学习...')
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)  # 批次
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch
    print('Epoch:', (epoch + 1), 'cost =', avg_cost)
print('学习完成')

# 测试模型检查准确率
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={X: mnist.test.images[:5000], Y: mnist.test.labels[:5000]}))

# 在测试集中随机抽一个样本进行测试
r = random.randint(0, mnist.test.num_examples - 1)
print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
print("Prediction: ", sess.run(tf.argmax(logits, 1), feed_dict={X: mnist.test.images[r:r + 1]}))

# 手写生成一个24位彩色bmp图片（宽度28，高度28）进行测试
img = plt.imread('1.bmp')
gravity = np.array([1., 0., 0.])
greyimg = np.dot(255 - img, gravity)/255
print("Prediction: ", sess.run(tf.argmax(logits, 1), feed_dict={X: greyimg.reshape([1, 784])}))
plt.imshow(greyimg, cmap='Greys', interpolation='nearest')
plt.show()
