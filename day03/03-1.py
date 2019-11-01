# -*- coding: utf-8 -*-
# @Time    : 2019/10/11 0011 10:40
# @Author  : 
# @FileName: 03-1.py
# @Software: PyCharm
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import random
import matplotlib.pylab as plt
import numpy as np
import cv2



# seed
tf.set_random_seed(111)
# data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
nb_classes = 10
# print(mnist.train.num_examples)
# print(mnist.train.next_batch(3))
# XYwb
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, nb_classes])
w = tf.Variable(tf.random_normal([784, nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes]))
# hypothesis
logits = tf.matmul(X, w) + b
hypothesis = logits
# cost
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
# train
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
# accuracy
predicted = tf.argmax(hypothesis, 1)
correct_pred = tf.equal(predicted, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    training_epochs = 15
    batch_size = 128
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, train], feed_dict={X:batch_xs,Y:batch_ys})
            avg_cost += c/total_batch
        print(epoch, avg_cost)
    # accuracy
    print('accuracy', sess.run(accuracy, feed_dict={X:mnist.test.images, Y:mnist.test.labels}))
    r = random.randint(0, mnist.test.num_examples - 1)
    print('Labels:', sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
    print('Prediction:', sess.run(tf.argmax(hypothesis, 1), feed_dict={X:mnist.test.images[r: r+1]}))
    plt.imshow(mnist.test.images[r].reshape(28,28), cmap='Greys')
    plt.show()
    # test
    # images = cv2.imread('./images1/1-1.PNG')
    # images = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
    # images = images.reshape((-1, 28*28))
    from aaa import load_images, load_images_labels
    images_data, img_labs = load_images()
    print('test_predicted:', sess.run(predicted, feed_dict={X:images_data}))