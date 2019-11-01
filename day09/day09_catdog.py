# -*- coding: utf-8 -*-
# @Time    : 2019/10/16 0016 14:58
# @Author  : 
# @FileName: day09_catdog.py
# @Software: PyCharm

import tensorflow as tf
import random
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import misc

tf.set_random_seed(111)
# data
all_num = 1000
split_num = int(all_num * 0.9)
train_num = split_num
test_num = all_num - train_num

IMGSIZE = 100


def get_all_files(file_path):
    image_list = []
    label_list = []
    cat_count = 0
    dog_count = 0
    for item in os.listdir(file_path):
        item_path = os.path.join(file_path, item)
        if item[:3] == 'cat':
            label_list.append([1, 0])
            cat_count += 1
        else:
            label_list.append([0, 1])
            dog_count += 1
        image_list.append(item_path)
    print('数据集中有%d只猫,%d只狗.' % (cat_count, dog_count))
    print(image_list[:5])
    # shuffle
    np.random.seed(111)
    shuffle_indices = np.random.permutation(np.arange(len(label_list)))
    x_shuffled = np.array(image_list)[shuffle_indices]
    y_shuffled = np.array(label_list)[shuffle_indices]
    image_list = x_shuffled[:train_num]
    label_list = y_shuffled[:train_num]
    image_test = x_shuffled[-test_num:]
    label_test = y_shuffled[-test_num:]
    return image_list, label_list, image_test, label_test


image_dir = r'E:\bawei\DeepLearning_1\深度一\tensorflow补充\catdog_data\data\train'
test_dir = r'E:\bawei\DeepLearning_1\深度一\tensorflow补充\catdog_data\data\test'
train_list, Y_one_hot, test_list, Y_test = get_all_files(image_dir)


# read image
def readimg(file):
    image = plt.imread(file)
    image = misc.imresize(image, (IMGSIZE, IMGSIZE))
    image = image / 255
    return image


"""
plt 读入图片需要图片路径， 得到的是三维数组


"""
i1 = readimg(train_list[44])
print(i1.shape)  # (100, 100, 3)
imgs = []
imgs_test = []
for i in range(0, train_num):
    image_train = readimg(train_list[i])
    imgs.append(image_train)
    if i % 100 == 0:
        print('read train', i)
for i in range(0, test_num):
    image_test = readimg(train_list[i])
    imgs_test.append(image_test)
    if i % 100 == 0:
        print('read test', i)
# train_data  test_data
imgArr = np.array(imgs)
imgArrTest = np.array(imgs_test)

g_b = 0


def next_batch(size):
    global g_b
    xb = imgArr[g_b:g_b + size]
    yb = Y_one_hot[g_b:g_b + size]
    g_b = g_b + size
    return xb, yb


# parameter
learning_rate = 0.0001
train_epochs = 100
batch_size = 100
# XYwb
X = tf.placeholder(tf.float32, [None, IMGSIZE, IMGSIZE, 3])
Y = tf.placeholder(tf.float32, [None, 2])
with tf.variable_scope('conv1'):
    w1 = tf.Variable(tf.random_normal([3, 3, 3, 16]), name='w1')
    L1 = tf.nn.conv2d(X, w1, strides=[1, 1, 1, 1], padding='SAME')
    L1 = tf.nn.relu(L1)
    L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
with tf.variable_scope('conv2'):
    w2 = tf.Variable(tf.random_normal([3, 3, 16, 32]), name='w2')
    L2 = tf.nn.conv2d(L1, w2, strides=[1, 2, 2, 1], padding='SAME')
    L2 = tf.nn.relu(L2)
    L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    dim = L2.get_shape()[1].value * L2.get_shape()[2].value * L2.get_shape()[3].value
    L2_flat = tf.reshape(L2, [-1, dim])
with tf.variable_scope("full_connect1"):
    w3 = tf.get_variable('w3', [dim, 128],
                         initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.Variable(tf.random_normal([128]), name='b3')
    L3 = tf.nn.relu(tf.matmul(L2_flat, w3) + b3)
with tf.variable_scope('full_connect2'):
    w4 = tf.get_variable('w4', [128, 128], initializer=tf.truncated_normal_initializer(stddev=0.005))
    b4 = tf.Variable(tf.random_normal([128]), name='b4')
    L4 = tf.nn.relu(tf.matmul(L3, w4) + b4)
    L4 = tf.nn.dropout(L4, keep_prob=0.9)
with tf.variable_scope('softMax'):
    w5 = tf.get_variable('w5', [128, 2], initializer=tf.truncated_normal_initializer(stddev=0.005))
    b5 = tf.get_variable('b5', [2], initializer=tf.constant_initializer(0.1))
    logits = tf.add(tf.matmul(L4, w5), b5)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(Y, 1), tf.argmax(logits, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(train_epochs):
        avg_cost = 0
        total_batch = int(train_num / batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = next_batch(batch_size)
            feed_dict = {X: batch_xs, Y: batch_ys}
            c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
            avg_cost += c / total_batch
        if epoch % 20 == 0:
            acc = sess.run(accuracy, feed_dict={X: imgArrTest, Y: Y_test})
            print('Epoch:', (epoch + 1), 'cost =', avg_cost, 'acc=', acc)
    print('学习完成')
    # 测试模型检查准确率
    print('Accuracy:', sess.run(accuracy, feed_dict={X: imgArrTest, Y: Y_test}))
