# -*- coding: utf-8 -*-
# @Time    : 2019/10/29 0029 14:01
# @Author  : 
# @FileName: 2.py
# @Software: PyCharm
# 利用tensorflow开发rnn网络实现手写数字识别，手写数据集文件夹名是MNIST_data
# (1)	引入必要的依赖库，读取加载MNIST数据集赋给mnist（5分）
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops.rnn import dynamic_rnn
mnist = input_data.read_data_sets("MNIST_data")
tf.set_random_seed(111)
# (2)	定义如下变量：时间步数n_steps=28,序列长度n_inputs=28, 隐藏状态n_neurons=100，输出分类n_outputs=10,学习率learn_rate=0.001,训练周期n_epochs=1,批大小batch_size=100，定义输入占位符:X,Y（5分）
# parameter
n_steps=28
n_inputs=28
n_neurons=100
n_outputs=10
learn_rate=0.001
n_epochs=1
batch_size=100
# placeholder
X = tf.placeholder(tf.float32, [None, 784])
X_ = tf.reshape(X, [-1, 28, 28])
Y = tf.placeholder(tf.int32, [None])
# (3)	定义模型预测：模型使用简单的BasicRNNcell,使用dynamic_rnn构建动态神经网络（5分）
cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=None)
outputs, last_state = dynamic_rnn(cell, X_, dtype=tf.float32)
# (4)	通过无激活函数的全连接层计算线性回归分类预测，赋值给logics；定义计算代价与损失，定义计算准确率（5分）
logics = tf.contrib.layers.fully_connected(outputs[:, -1], num_outputs=n_neurons, activation_fn=None)
# cost
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logics, labels=Y))
# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(cost)
# accuracy
predicted = tf.nn.in_top_k(logics, Y, 1)
accuracy = tf.reduce_mean(tf.cast(predicted, tf.float32))
# (5)	定义session 并初始化，迭代训练，训练过程打印训练结果（5分）
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(n_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)
        for i in range(total_batch):
            xs, ys = mnist.train.next_batch(batch_size)
            xs.reshape((-1, 28, 28))
            feed_dict = {X: xs, Y: ys}
            l, _, acc = sess.run([cost, optimizer, accuracy], feed_dict=feed_dict)
            avg_cost += l / total_batch
        print("avg_cost:  ", avg_cost, "accuracy:   ", acc)
# (6)	运行成功有结果（5分）
