# -*- coding: utf-8 -*-
# @Time    : 2019/10/22 0022 8:36
# @Author  : 
# @FileName: test15.py
# @Software: PyCharm

# 使用tensorflow框架，利用卷积神经网络实现手写数字识别功能。
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import random
import matplotlib.pyplot as plt
# (一)	读取tensorflow自带的mnist数据集（8分）
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
# (二)	定义占位符（8分）
X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])
# (三)	定义第1个卷积层，卷积（6分）、relu（3）、池化（5分）
w1 = tf.Variable(tf.random_normal([3,3,1,32]))
L1 = tf.nn.conv2d(X_img, w1, strides=[1,1,1,1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
# (四)	定义第2个卷积层，卷积（6分）、relu（3）、池化（5分）
w2 = tf.Variable(tf.random_normal([3,3,32,64]))
L2 = tf.nn.conv2d(L1, w2, strides=[1,1,1,1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
# (五)	定义全连接层（8分）
dim = L2.get_shape()[1].value*L2.get_shape()[2].value*L2.get_shape()[3].value
L2_flat = tf.reshape(L2, [-1, dim])
w3 = tf.get_variable('w3', [dim, 10], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]))
logits = tf.matmul(L2_flat, w3) + b
# (六)	定义代价/损失函数（8分）
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
# (七)	训练迭代过程（8分）
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
train_epochs =12
batch_size = 128
prediction = tf.argmax(logits, 1)
correct_pred = tf.equal(prediction, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# (八)	对训练集分批次训练（8分）
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(train_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            xs, ys = mnist.train.next_batch(batch_size)
            feed_dict = {X:xs, Y:ys}
            c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
# (九)	计算每批次的平均损失值（8分）
            avg_cost += c/total_batch
# (十)	准确率计算（8分）
    print("accuracy:", sess.run(accuracy, feed_dict={X:mnist.test.images, Y:mnist.test.labels}))
# (十一)	从测试集中随机抽取一个样本进行测试验证（8分）
    r = random.randint(0, mnist.test.num_examples-1)
    print("accuracy:", sess.run(accuracy, feed_dict={X: mnist.test.images[r:r+1], Y: mnist.test.labels[r:r+1]}))
    plt.imshow(mnist.test.images[r:r+1].reshape((28, 28)), cmap='Greys')
    plt.show()