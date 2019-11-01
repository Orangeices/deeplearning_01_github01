# -*- coding: utf-8 -*-
# @Time    : 2019/10/17 0017 8:52
# @Author  : 
# @FileName: 09_11_file.py
# @Software: PyCharm

# 从文本文件中获取图片数据和标签
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import numpy as np

tf.set_random_seed(111)
imgArr = np.loadtxt(r'E:\bawei\DeepLearning_1\深度一\tensorflow补充\images.txt', delimiter=',')/255
total = imgArr.shape[0]
Y_one_hot = np.eye(10)[np.loadtxt(r'E:\bawei\DeepLearning_1\深度一\tensorflow补充\labels.txt',
                                  dtype=int).tolist()]
train_test = 0.8

g_b = 0
def next_batch(size):
    global g_b
    xb = imgArr[g_b:g_b+size]
    yb = Y_one_hot[g_b:g_b+size]
    g_b += size
    return xb, yb

print(imgArr.shape)
print([np.loadtxt(r'E:\bawei\DeepLearning_1\深度一\tensorflow补充\labels.txt',
                                  dtype=int).tolist()])
# parameter
learning_rate = 0.001
train_epochs = 15
batch_size = 100

# XYwb
X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28,28,1])
Y = tf.placeholder(tf.float32, [None, 10])

# layers1
with tf.variable_scope('conv1'):
    w1 = tf.Variable(tf.random_normal([3, 3, 1, 32]), name='w1')
    L1 = tf.nn.conv2d(X_img, w1,strides=[1,1,1,1], padding='SAME')
    L1 = tf.nn.relu(L1)
    L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# layers2
with tf.variable_scope('conv2'):
    w2 = tf.Variable(tf.random_normal([3, 3, 32, 64]), name='w2')
    L2 = tf.nn.conv2d(L1, w2,strides=[1,1,1,1], padding='SAME')
    L2 = tf.nn.relu(L2)
    L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


    dim = L2.get_shape()[1].value * L2.get_shape()[2].value* L2.get_shape()[3].value
    L2_flat = tf.reshape(L2, [-1, dim])

# full_connect
with tf.variable_scope('full_connect'):
    w3 = tf.get_variable('w3', [dim, 10], initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.random_normal([10]), name='b')

logits = tf.matmul(L2_flat, w3) + b

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#accuracy
predicted = tf.argmax(logits, 1)
correct_pred = tf.equal(predicted, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
#
saver = tf.train.Saver({"conv1/w1":w1, "conv2/w2":w2, "full_connect/w3":w3, "full_connect/b":b})
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # for epoch in range(train_epochs):
    #     avg_cost = 0
    #     total_batch = int(total*train_test / batch_size)
    #     g_b = 0
    #     for i in range(total_batch):
    #         batch_xs, batch_ys = next_batch(batch_size)
    #         c, _ = sess.run([cost, optimizer], feed_dict={X:batch_xs, Y:batch_ys})
    #         avg_cost += c / total_batch
    #     print('epoch:', (epoch+1), 'cost', avg_cost)
    # print('学习完成')
    # saver.save(sess, './ckpt/cnn_file')
    saver.restore(sess, './ckpt/cnn_file')
    batch_xs, batch_ys = next_batch(batch_size)
    c = sess.run(cost, feed_dict={X:batch_xs, Y:batch_ys})
    print(c)
    print('Accuracy:',
          sess.run(accuracy, feed_dict={X: imgArr[int(total * train_test):], Y: Y_one_hot[int(total * train_test):]}))
    #
    img = plt.imread(r'E:\bawei\DeepLearning_1\深度一\tensorflow补充\1.bmp')
    gravity = np.array([1.,0.,0.])
    greyimg = np.dot(255-img, gravity)
    p = sess.run(tf.argmax(logits, 1), feed_dict={X:greyimg.reshape([1,784])})
    print(p)
    plt.imshow(greyimg, cmap='Greys', interpolation='nearest')
    plt.show()