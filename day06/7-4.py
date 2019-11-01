# -*- coding: utf-8 -*-
# @Time    : 2019/10/11 0011 20:15
# @Author  : 
# @FileName: 7-4.py
# @Software: PyCharm


# import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
# import random
# tf.set_random_seed(111)
# # data
# mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
#
# # parameter
# nb_classes = 10
# # XYwb
# X = tf.placeholder(tf.float32, [None, 784])
# Y = tf.placeholder(tf.float32, [None, nb_classes])
# w = tf.Variable(tf.random_normal([784, nb_classes]))
# b = tf.Variable(tf.random_normal([nb_classes]))
# # hypothesis
# logits = tf.matmul(X, w) + b
# # cost
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
# # train
# train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
# # accuracy
# predicted = tf.argmax(logits, 1)
# correct_pred = tf.equal(tf.argmax(Y, 1), predicted)
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
#
# saver = tf.train.Saver()
#
# # session
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     batch_size = 128
#     train_epochs = 15
#     total_batch = int(mnist.train.num_examples / batch_size)
#     for epoch in range(train_epochs):
#         avg_cost = 0
#         for i in range(total_batch):
#             batch_xs, batch_ys = mnist.train.next_batch(batch_size)
#             c, _ = sess.run([cost, train], feed_dict={X:batch_xs, Y:batch_ys})
#             avg_cost += c/total_batch
#         print(avg_cost)
#     r = random.randint(0, mnist.test.num_examples-1)
#     p = sess.run(tf.argmax(logits, 1), feed_dict={X:mnist.test.images[r:r+1]})
#     print(p)
#     print(sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
#     plt.imshow(mnist.test.images[r].reshape(28, 28), cmap='Greys')
#     plt.show()