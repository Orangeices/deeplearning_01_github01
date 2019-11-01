# -*- coding: utf-8 -*-
# @Time    : 2019/10/18 0018 21:02
# @Author  : 
# @FileName: practice04.py
# @Software: PyCharm
# ①　导入Tensorflow模块。（6分）
# import tensorflow as tf
# # ②　导入Tensorflow里面的mnist数据集。（7分）
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
# # ⑦　定义网络输入的占位符xs和ys，以及dropout的的占位符。（7分）
# X = tf.placeholder(tf.float32, [None, 784])
# Y = tf.placeholder(tf.float32, [None, 10])
# # ⑧　将xs的形状变成[-1,28,28,1]。（7分）
# xs = tf.reshape(X, [-1, 28, 28, 1])
# # ③　定义Weight变量，输入shape，返回变量的参数（使用了tf.truncted_normal产生随机变量来进行初始化）。（6分）
# w = tf.Variable(tf.truncated_normal([3,3,1,10]), name='w')
# # ④　定义biase变量，输入shape，返回变量的一些参数（使用tf.constant常量函数来进行初始化）。（6分）
# b = tf.Variable(tf.truncated_normal([10]), name='b')
# # ⑤　定义卷积操作；其中步长strides=[1, 1, 1, 1]，padding采用的方式是“SAME”。（7分）
# L = tf.nn.conv2d(xs, w, strides=[1,1,1,1], padding="SAME")
# L = tf.nn.relu(L)
# # ⑥　定义池化操作；其中ksize=[1, 2, 2, 1]，步长为2，padding采用的方式是“SAME”。（7分）
# L = tf.nn.max_pool(L, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
# L = tf.nn.dropout(L, keep_prob=0.9)
# # ⑨　按上图所示建立卷积层。（7分）
#
# # ⑩　按上图所示建立全连接层。（7分）
# dim = L.get_shape()[1].value*L.get_shape()[2].value*L.get_shape()[3].value
# L_flat = tf.reshape(L, [-1, dim])
# w1 = tf.Variable(tf.random_normal([dim, 10]), name='w1')
# b1 = tf.Variable(tf.random_normal([10]), name='b1')
#
# # 11　构建最后一层：输出层；然后用softmax分类器（多分类，输出的是各个类的概率），对我们的输出进行分类。（7分）
# logits = tf.matmul(L_flat, w1) + b1
#
# # 12　利用交叉熵损失函数来定义cost function。（7分）
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
# # 13　用tf.train.AdamOptimizer()作为优化器进行优化。（6分）
# train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(cost)
#
# predicted = tf.argmax(logits, 1)
# correct_pred = tf.equal(predicted, tf.argmax(Y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# # 14　定义session，并初始化所有变量。（7分）
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#
#     train_epochs = 12
#     batch_size = 128
#     total_batch = int(mnist.train.num_examples/batch_size)
#     for epoch in range(train_epochs):
#         avg_cost = 0
#         for i in range(total_batch):
#             batch_xs, batch_ys = mnist.train.next_batch(batch_size)
#             c, _ = sess.run([cost, train], feed_dict={X:batch_xs, Y:batch_ys})
#
#         avg_cost += c / total_batch
#         print(avg_cost)
# # 15　训练1000次，每个50次检查一下模型的精度。（6分）
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

tf.set_random_seed(111)

X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 1])

# layer1
w1 = tf.Variable(tf.random_normal([3,3,1,32]))
L1 = tf.nn.conv2d(X_img, w1, strides=[1,1,1,1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
# layer2
w2 = tf.Variable(tf.random_normal([3,3,32,64]))
L2 = tf.nn.conv2d(L1, w2, strides=[1,1,1,1], padding='SAME')
l2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
#fc1
dim = L2.get_shape()[1].value*L2.get_shape()[2].value*L2.get_shape()[3].value
L2_flat = tf.reshape(L2, [-1, dim])
w3 = tf.get_variable('w3',[dim, 128],
                     initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([128]), name='b3')
L3 = tf.nn.relu(tf.matmul(L2_flat, w3)+b3)
#fc2
w4 = tf.get_variable('w4', [128, 128],
                     initializer=tf.truncated_normal_initializer(stddev=0.005))
b4 = tf.get_variable('b4', [128],initializer=tf.constant_initializer(0.1))
L4 = tf.nn.relu(tf.matmul(L3, w4)+b4)
L4 = tf.nn.dropout(L4, keep_prob=0.9)
# fc3
w5 = tf.get_variable('w5', [128, 10],
                     initializer=tf.truncated_normal_initializer(0.005))
b5 = tf.get_variable('b5', [10],initializer=tf.constant_initializer(0.1))
logits = tf.add(tf.matmul(L4, w5), b5)