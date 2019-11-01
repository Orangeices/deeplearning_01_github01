# -*- coding: utf-8 -*-
# @Time    : 2019/10/13 0013 20:44
# @Author  : 
# @FileName: Cifar_task.py
# @Software: PyCharm
import _pickle as cPickle
import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf
import os
CIFAR_DIR = r'F:\神经网络CNN RNN GAN-mp4\神经网络常用算法\cifar-10-batches-py'
print(os.listdir(CIFAR_DIR))
def load_data(filename):
    """读取数据"""
    with open(os.path.join(CIFAR_DIR, 'data_batch_1'), 'rb') as f:
        data = cPickle.load(f, encoding='iso-8859-1')
        # print(type(data))
    return data['data'], data['labels']

# 32*32 = 1024 * 3 = 3072
# RGB
# image_arr = data['data'][100]
# print(image_arr.shape)
# image_arr = image_arr.reshape((3, 32, 32))
# image_arr = image_arr.transpose((1, 2, 0))
# plt.imshow(image_arr)
# plt.show()
x = tf.placeholder(tf.float32, [None, 3072])
y = tf.placeholder(tf.int64, [None])

w = tf.Variable(tf.random_normal([3072, 1]), name='w')
b = tf.Variable(tf.random_normal([1]), name='b')

# hypothesis
logits = tf.matmul(x, w)+b
hypothesis = tf.nn.sigmoid(logits)

#


