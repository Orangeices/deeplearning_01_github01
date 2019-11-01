# -*- coding: utf-8 -*-
# @Time    : 2019/10/29 0029 13:39
# @Author  : 
# @FileName: 1.py
# @Software: PyCharm
# (1)	导入程序要用到的所有库文件并设置随机种子。（5分）
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import random
import numpy as np

tf.set_random_seed(111)
# (2)	编写程序加载衣服数据集。（5分）
mnist = input_data.read_data_sets("MNIST_data1", one_hot=True)
clothe = ['T-shirt','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle']
print(mnist.test.images[0].shape)  # (784,)
# (3)	定义三个变量：学习率（learning_rate,学习率为0.001）,
# 学习周期（traing_epochs，学习周期为15）,批大小（batch_size批大小为100）（5分）
learning_rate = 0.001
train_epochs = 15
batch_size = 100
# (4)	定义占位符X,Y（6分）
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])
# (5)	定义权重和偏置：W,b（6分）
W = tf.Variable(tf.random_normal([784, 10]))
b = tf.Variable(tf.random_normal([10]))
# (6)	编写神经网络预测模型（8分）
logits = tf.matmul(X, W) + b
# (7)	编写代价或损失函数。（7分）
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
# (8)	编写网络优化函数，采用adam。（6分）
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# (9)	创建会话与全局变量初始化。（6分）
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(train_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            xs, ys = mnist.train.next_batch(batch_size)
            feed_dict = {X: xs, Y: ys}
            l, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
            avg_cost += l/total_batch
        print(avg_cost)
# (10)	迭代训练，循环训练周期为20，batch_size大小设置100，每批打印输出一次结果。（6分）
# (11)	在测试集中随机抽取一个样本进行测试并输出测试结果（含衣服的名字）（5分）
    r = random.randint(0, mnist.test.num_examples-1)
    print("prediction:  ", sess.run(tf.argmax(logits, 1), feed_dict={X:mnist.test.images[r: r+1]}))
    label = np.argmax(mnist.test.labels[r: r + 1], 1)
    print("label:   ", clothe[label[0]])

# (12)	运行成功有结果（5分）
