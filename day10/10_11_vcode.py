# -*- coding: utf-8 -*-
# @Time    : 2019/10/17 0017 21:10
# @Author  : 
# @FileName: 10_11_vcode.py
# @Software: PyCharm
import tensorflow as tf
import random
import os
import numpy as np
from PIL import Image
tf.set_random_seed(111)

train_num = 1000
test_num = 100

IMG_HEIGHT = 60
IMG_WIDTH = 160
char_num = 4
characters = range(10)
labellen = char_num * len(characters)

def label2vec(label):
    label_vec = np.zeros(char_num * len(characters))
    for i, num in enumerate(label):
        idx = i * len(characters) + int(num)
        label_vec[idx] = 1
    return label_vec

# print(label2vec('1234'))

def convert2gray(img):
    if len(img.shape)>2:
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img
def get_all_files(file_path, num):
    image_list = []
    label_list = []
    i = 0
    for item in os.listdir(file_path):
        item_path = os.path.join(file_path, item)
        image = Image.open(item_path)
        image = convert2gray(np.array(image))
        image_array = np.array(image) /255.0
        image_list.append(image_array.reshape(IMG_HEIGHT,IMG_WIDTH, 1))
        label = os.path.splitext(os.path.split(item)[1])[0]
        label_list.append(label2vec(label))
        i += 1
        if i >= num:
            break
    return np.array(image_list), np.array(label_list)

path = r'E:\bawei\DeepLearning_1\深度一\tensorflow补充\vcode_data'
image_dir = path+r'\train'
test_dir = path+r'\test'
imgArr, Y_one_hot = get_all_files(image_dir, train_num)
imgArrTest, Y_test = get_all_files(test_dir, test_num)
print(imgArr.shape)


g_b = 0
def next_batch(size):
    global g_b
    xb = imgArr[g_b:g_b+size]
    yb = Y_one_hot[g_b:g_b+size]
    g_b = g_b + size
    return xb, yb

# parameter
learning_rate = 0.001
training_epochs = 100
batch_size = 100

X = tf.placeholder(tf.float32, [None, IMG_HEIGHT, IMG_WIDTH, 1])
Y = tf.placeholder(tf.float32, [None, labellen])

w1 = tf.Variable(tf.random_normal([3,3,1,32]), name='w1')
L1 = tf.nn.conv2d(X, w1, strides=[1,1,1,1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

w2 = tf.Variable(tf.random_normal([3,3,32,64], stddev=0.01), name='w2')
L2 = tf.nn.conv2d(L1, w2, strides=[1,1,1,1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

dim = L2.get_shape()[1].value*L2.get_shape()[2].value*L2.get_shape()[3].value
L2_flat = tf.reshape(L2, [-1, dim])
with tf.variable_scope('fc1'):
    w3 = tf.get_variable('w3', [dim, 128],
                         initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.Variable(tf.random_normal([128]))
    L3 = tf.nn.relu(tf.matmul(L2_flat, w3) + b3)
with tf.variable_scope('fc2'):
    w4 = tf.get_variable('w4', [128, 128],
                         initializer=tf.truncated_normal_initializer(stddev=0.005))
    b4 = tf.get_variable('b4', [128],
                         initializer=tf.constant_initializer(0.1))
    L4 = tf.nn.relu(tf.matmul(L3, w4) + b4)
    L4 = tf.nn.dropout(L4, keep_prob=0.9)
with tf.variable_scope('f3'):
    w5 = tf.get_variable('w5', [128, labellen],
                         initializer=tf.truncated_normal_initializer(stddev=0.005))
    b5 = tf.get_variable('b5', [labellen],
                         initializer=tf.constant_initializer(0.1))
    logits = tf.add(tf.matmul(L4, w5), b5)

cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=Y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# accuracy
predict = tf.reshape(logits, [-1, 4, 10])
correct_pred = tf.equal(tf.argmax(predict, 2),
                        tf.argmax(tf.reshape(Y, [-1, 4, 10]), 2))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# saver
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(train_num/batch_size)
        g_b = 0
        for i in range(total_batch):
            batch_xs, batch_ys = next_batch(batch_size)
            feed_dict = {X:batch_xs, Y:batch_ys}
            c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
            avg_cost += c/total_batch
        if epoch%20 == 0:
            acc = sess.run(accuracy, feed_dict={X:imgArrTest, Y:Y_test})
            print('Epoch:', (epoch + 1), 'cost =', avg_cost, 'acc=', acc)
    print('学习完成')
    # saver.save(sess, './ckpt/cnn_vcode')
    r = random.randint(0, test_num-1)
    print("label:", sess.run(tf.argmax(logits, 1), feed_dict={X:imgArrTest[r:r+1]}))
