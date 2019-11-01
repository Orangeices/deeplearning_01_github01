# -*- coding: utf-8 -*-
# @Time    : 2019/10/16 0016 8:36
# @Author  : 
# @FileName: test10.py
# @Software: PyCharm
# 使用tensorflow自带的mnist数据集，用softmax多分类方式实现手写数字识别。
# (一)	导入tensorflow模块，设置随机种子（6分）
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
tf.set_random_seed(111)
# (二)	读取tensorflow自带的mnist数据集（8分）
mnist = input_data.read_data_sets("MNIST_date", one_hot=True)
# parameter
learning_rate = 0.01
train_epochs = 15
batch_size = 128
total_batch = int(mnist.train.num_examples/batch_size)
nb_classes = 10
# (三)	定义占位符（8分）
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, nb_classes])
# (四)	定义权重和偏置（8分）
w = tf.Variable(tf.random_normal([784, nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes]))
# (五)	定义预测模型（8分）
logits = tf.matmul(X, w) + b
# (六)	定义代价或损失函数（8分）
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
# (七)	使用梯度下降优化器（8分）
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# accuracy
predicted = tf.argmax(logits, 1)
correct_pred = tf.equal(predicted, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# (八)	创建会话、全局变量初始化（6分）
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # (九)	大周期15次，进行迭代训练（8分）
    for epoch in range(train_epochs):
        avg_cost = 0
        # (十)	对训练集分批次训练（8分）
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            cost_val, _ ,acc = sess.run([cost, optimizer, accuracy],
                                        feed_dict={X:batch_xs, Y:batch_ys})
            # (十一)	计算每批次的平均损失值（8分）
            avg_cost += cost_val/total_batch
        print(epoch, cost_val, "准确率:", acc)  # (十二)	准确率计算（8分）
    import random
    r = random.randint(0, mnist.test.num_examples-1)
    p1 = sess.run(tf.argmax(logits, 1), feed_dict={X:mnist.test.images[r:r+1],
                                                   Y:mnist.test.labels[r:r+1]})

# (十三)	从测试集中随机抽取一个样本进行测试验证（8分）

