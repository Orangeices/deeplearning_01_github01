# -*- coding: utf-8 -*-
# @Time    : 2019/10/21 0021 8:36
# @Author  : 
# @FileName: week03.py
# @Software: PyCharm
# (1)	导入必要的依赖库。（8分）
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# (2)	设置随机种子（8分）
tf.set_random_seed(111)
# (3)	读取加载MNIST数据集，赋值给变量mnist（8分）
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
print(mnist.train.images[0:1].shape)
# (4)	定义学习率learning_rate为0.001（8分）
learning_rate = 0.001
# (5)	定义学习周期training_epochs为15（8分）
training_epochs = 15
# (6)	定义批大小batch_size为100（8分）
batch_size = 100
# (7)	定义权重与偏执变量W,b（8分）
w = tf.Variable(tf.random_normal([784, 10]))
b = tf.Variable(tf.random_normal([10]))
# (8)	定义输入占位符X。（8分）
X = tf.placeholder(tf.float32, [None, 784])
# (9)	定义输出占位符Y。（8分）
Y = tf.placeholder(tf.float32, [None, 10])
# (10)	定义线性激活函数进行预测。（8分）
logits = tf.nn.relu(tf.matmul(X, w) + b)
# (11)	定义模型输出与交叉熵损失函数（4分）
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
# (12)	模型训练，利用优化函数进行优化，优化方法采用adam（4分）
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# (13)	创建会话（4分）
with tf.Session() as sess:
# (14)	全局变量初始化（4分）
    sess.run(tf.global_variables_initializer())
# (15)	迭代训练,每个训练周期输出一次loss值（4分）
    for epoch in range(training_epochs):
        avg_loss = 0
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            xs, ys = mnist.train.next_batch(batch_size)
            feed_dict = {X:xs, Y:ys}
            loss, _ = sess.run([cost, optimizer],feed_dict=feed_dict)
            avg_loss += loss/total_batch
        print('epoch:', epoch+1, "  loss:", loss)

