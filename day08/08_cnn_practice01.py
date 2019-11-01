# -*- coding: utf-8 -*-
# @Time    : 2019/10/14 0014 20:01
# @Author  : 
# @FileName: 08_cnn_practice01.py
# @Software: PyCharm
# import tensorflow as tf
# import random
# import matplotlib.pylab as plt
# import numpy as np
# from tensorflow.examples.tutorials.mnist import input_data
#
# tf.set_random_seed(111)
# mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
# # parameter
# learning_rate = 0.001
# train_epochs = 15
# batch_size = 100
# # XYwb
# X = tf.placeholder(tf.float32, [None, 784])
# X_img = tf.reshape(X, [-1, 28, 28, 1])  # (28, 28, 1)
# Y = tf.placeholder(tf.float32, [None, 10])
#
#
# # layer1
# """
# 对于卷积来说，两者作用是一样的。最大的区别在于：
#
# layers中     filters：一个整数，代表滤波器的个数，也就是卷积核数或者输出空间的维数
# nn中         filter：  一个四维张量，[filter_height, filter_width, in_channels, out_channels]
# 相当于nn中的filter在layers中被拆成了filters和kernel_size两个参数。
#
# 此外：
#
# layers更适合从头到尾训练模型，因为activation和bias自动实现，而nn则需要你显示地创建placeholder并进行相关计算
#
# nn更适合加载已经预训练好的模型，因为filter由tf.Variable生成，在加载预训练的权值时更快
# nn可以用权重衰减（因为权重矩阵是你显示定义的，layers中根据参数为你自动定义weights）如下：
# kernel = _variable_with_weight_decay('weights',
#                                          shape=[5, 5, 3, 64],
#                                          stddev=5e-2,
#                                          wd=0.0)
# conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
#
# ————————————————
# 版权声明：本文为CSDN博主「sxc的csdn」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
# 原文链接：https://blog.csdn.net/zongza/article/details/85099334
#
#
# """
# # tf.layers.conv2d()
# w1 = tf.Variable(tf.random_normal([3, 3, 1 ,32]))
# L1 = tf.nn.conv2d(X_img, w1, strides=[1,1,1,1], padding='SAME') # (?, 28, 28, 32)
# L1 = tf.nn.relu(L1)
# L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') # (?, 14, 14, 32)
# # layer2
# w2 = tf.Variable(tf.random_normal([3, 3, 32, 64]))
# L2 = tf.nn.conv2d(L1, w2, strides=[1,1,1,1], padding='SAME')
# L2 = tf.nn.relu(L2)
# L2 = tf.nn.max_pool(L2, ksize=[1,2,2, 1], strides=[1,2,2,1], padding='SAME')
# L2_flat = tf.reshape(L2, [-1, 7*7*64])  # 变成一维向量 (?, 3136)
#
# # full_content
# w3 = tf.get_variable('w3', [7*7*64, 10],
#                      initializer=tf.contrib.layers.xavier_initializer())
# b = tf.Variable(tf.random_normal([10]))
# logits = tf.matmul(L2_flat, w3) + b
#
# # cost
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
# # optimizer
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#
#
# # accuracy
# predicted = tf.argmax(logits, 1)
# correct_pred = tf.equal(predicted, tf.argmax(Y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
#
# # save
# saver = tf.train.Saver()
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     # for epoch in range(train_epochs):
#     #     avg_cost = 0
#     #     total_batch = int(mnist.train.num_examples/batch_size)
#     #     for i in range(total_batch):
#     #         batch_xs, batch_ys = mnist.train.next_batch(batch_size)
#     #         cost_val, _ = sess.run([cost, optimizer], feed_dict={X:batch_xs, Y:batch_ys})
#     #         avg_cost += cost_val/total_batch
#     #     print(avg_cost)
#     # print('over!~')
#     # saver.save(sess, './ckpt/cnn')
#     saver.restore(sess, './ckpt/cnn')
#     r = random.randint(0, mnist.train.num_examples-1)
#     _, acc, label, prediction = sess.run([optimizer,
#                                           accuracy,
#                                           tf.argmax(mnist.test.labels[:5], 1),
#                                           tf.argmax(logits, 1)],
#                                           feed_dict={X:mnist.test.images[:5], Y:mnist.test.labels[:5]})
#     print(acc, '\n',  label, '\n', prediction)

import tensorflow as tf
import matplotlib.pylab as plt
import random
from tensorflow.examples.tutorials.mnist import input_data
tf.set_random_seed(111)
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# parameter
learning_rate = 0.001
train_epochs = 15
batch_size = 100
total_batch = int(mnist.train.num_examples/batch_size)
nb_classes = 10

# XYwb
X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, nb_classes])

# layer1
with tf.variable_scope('conv1'):
    w1 = tf.Variable(tf.random_normal([3, 3, 1, 32]), name='w1')
    L1 = tf.nn.conv2d(X_img, w1, strides=[1,1,1,1], padding='SAME')
    L1 = tf.nn.relu(L1)
    L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# layer2
with tf.variable_scope('conv2'):
    w2 = tf.Variable(tf.random_normal([3, 3, 32, 64]), name='w2')
    L2 = tf.nn.conv2d(L1, w2, strides=[1,1,1,1], padding='SAME')
    L2 = tf.nn.relu(L2)
    L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2,2,1], padding='SAME')
    L2_flat = tf.reshape(L2, [-1, 7*7*64])

# full_connect
with tf.variable_scope('fc'):
    w3 = tf.get_variable('w3', [7*7*64, nb_classes],
                         initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.random_normal([nb_classes]))
logits = tf.matmul(L2_flat, w3) + b

# cost
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# accuracy
predicted = tf.argmax(logits, 1)
correct_pred = tf.equal(predicted, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
save = tf.train.Saver({"Variable":w1, 'Variable_1':w2, "w3":w3, 'Variable_2':b})
import os
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # if os.path.exists('./ckpt/checkpoint'):
    #     save.restore(sess, './ckpt/cnn')
    # else:l
    save.restore(sess, '.\ckpt\cnn')
    # loader = tf.train.import_meta_graph('./ckpt/cnn.meta')
    # loader.restore(sess, tf.train.latest_checkpoint('./ckpt'))
    # for epoch in range(train_epochs):
    #     avg_cost = 0
    #     for i in range(total_batch):
    #         batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    #         c, _ = sess.run([cost, optimizer], feed_dict={X:batch_xs, Y:batch_ys})
    #         avg_cost += c / total_batch
    #     print('Epoch:', (epoch + 1), 'cost =', avg_cost)

    print('Accuracy:', sess.run(accuracy,
                                feed_dict={X: mnist.test.images[:50],
                                           Y: mnist.test.labels[:50]}))
    # avg_cost = 0
    # batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    # c, _ = sess.run([cost, optimizer], feed_dict={X:batch_xs, Y:batch_ys})
    # avg_cost += c / total_batch
    # print(avg_cost)
    import numpy as np
    img = plt.imread(r'E:\bawei\DeepLearning_1\深度一\tensorflow补充\1.bmp')
    graity = np.array([1., 0., 0.])
    greyimg = np.dot(255-img, graity)/255
    print(sess.run(tf.argmax(logits, 1), feed_dict={X:greyimg.reshape(1,784)}))
    plt.imshow(greyimg, cmap='Greys')
    plt.show()