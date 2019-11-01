# -*- coding: utf-8 -*-
# @Time    : 2019/10/18 0018 14:37
# @Author  : 
# @FileName: 10_11_planecarbird.py
# @Software: PyCharm

import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
import random

np.random.seed(111)


def readImg(file):
    image = plt.imread(file)
    image = image / 255
    return image


images = []
labels = []

path = r'E:\bawei\DeepLearning_1\深度一\tensorflow补充\data3\data3'
for f in os.listdir(path):
    img = readImg(path + '\\' + f)
    images.append(img)
    ones = np.eye(3)
    labels.append(ones[int(f[0])])
images_data = np.array(images)
labels = np.array(labels)
print(images_data.shape)
# print(labels)
# shuffle
order = np.random.permutation(images_data.shape[0])
imgArr = images_data[order]
Y_one_hot = labels[order]
total = imgArr.shape[0]
train_test = 0.9

g_b = 0


def next_batch(size):
    global g_b
    xb = imgArr[g_b:g_b + size]
    yb = Y_one_hot[g_b:g_b + size]
    g_b += size
    return xb, yb


learning_rate = 0.001
training_epochs = 40
batch_size = 10

# XYwb
X = tf.placeholder(tf.float32, [None, 32, 32, 3])
Y = tf.placeholder(tf.float32, [None, 3])
