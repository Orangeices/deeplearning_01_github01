# -*- coding: utf-8 -*-
# @Time    : 2019/10/11 0011 20:45
# @Author  : 
# @FileName: 06_cnn_01.py
# @Software: PyCharm
import tensorflow as tf
import random
import matplotlib.pylab as plt
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

path = r'E:\bawei\DeepLearning_1\深度一\tensorflow补充'
# seed

tf.set_random_seed(111)
# data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
# parameter
learning_rate = 0.001
train_epochs = 2
batch_size = 100
nb_classes = 10
# XYwb
X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1]) # 灰图  [batch, in_height, in_width, in_channels]
Y = tf.placeholder(tf.float32, [None, nb_classes])
# layer1
w1 = tf.Variable(tf.random_normal([3, 3, 1, 32])) # 卷积核（3， 3）， 输入通道 1， 输出通道 32
#                  [filter_height, filter_width, in_channels, out_channels]
L1 = tf.nn.conv2d(X_img, w1, strides=[1, 1, 1, 1], padding='SAME') # 卷积输出（?, 28, 28, 32）
#      def conv2d(input, filter, strides, padding, use_cudnn_on_gpu=True, data_format="NHWC", dilations=[1, 1, 1, 1], name=None):
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME") # 池化输出（？， 14， 14， 32）
# layer2
w2 = tf.Variable(tf.random_normal([3, 3, 32, 64]))
L2 = tf.nn.conv2d(L1, w2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

L2_flat = tf.reshape(L2, [-1, 7*7*64]) # 变成一维向量

# 全连接
w3 = tf.get_variable('w3', shape=[7*7*64, 10],
                     initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]))
logits = tf.matmul(L2_flat, w3) + b

# cost

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
# train
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# accuracy
predicted = tf.argmax(logits, 1)
correct_pred = tf.equal(predicted, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.9 # 占用GPU90%的显存
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# 迭代
for epoch in range(train_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X:batch_xs, Y:batch_ys}
        c, _, = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c/total_batch
    print('epoch:', (epoch+1), 'cost:', avg_cost)
print('学习完成！~')
# test
print('Accuracy:', sess.run(accuracy, feed_dict={X: mnist.test.images[:5000], Y: mnist.test.labels[:5000]}))
# 在测试集中随机抽一个样本进行测试
r = random.randint(0, mnist.test.num_examples - 1)
print('Label:', sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
print('Prediction:', sess.run(tf.argmax(logits, 1), feed_dict={X:mnist.test.images[r:r+1]}))
# 手写生成一个24位彩色bmp图片（28， 28）
img = plt.imread(path+r'\1.bmp')
gravity = np.array([1., 0., 0.])
greyimg = np.dot(255-img, gravity)/255
print("Prediction: ", sess.run(tf.argmax(logits, 1), feed_dict={X: greyimg.reshape([1, 784])}))
plt.imshow(greyimg, cmap='Greys', interpolation='nearest')
plt.show()

sess.close()
