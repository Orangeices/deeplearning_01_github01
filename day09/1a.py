# -*- coding: utf-8 -*-
# @Time    : 2019/10/17 0017 14:17
# @Author  : 
# @FileName: 1a.py
# @Software: PyCharm
# import tensorflow as tf
#
# a = tf.constant([[
#     [
#         [1,3,5,7],
#         [8,6,4,2],
#         [4,2,8,6],
#         [1,3,5,7]
#     ],
#     [
#         [2,4,6,8],
#         [7,5,3,1],
#         [3,1,7,5],
#         [2,4,6,8]
#     ]
# ]])
#
# a = tf.reshape(a, [-1, 4, 4, 2])
# pooling = tf.nn.max_pool(a, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
# with tf.Session() as sess:
#     print("image:")
#     image = sess.run(a)
#     print(image.shape)
#     print(image)
#     print("reslut:")
#     result = sess.run(pooling)
#     print (result.shape)
#     print (result)

import numpy as np
import matplotlib.pyplot as plt
import os
np.random.seed(111)
train_test = 0.8
def get_filePath(file_path):
    imageDirs_list = []
    labels_list = []
    for f in os.listdir(file_path):
    #     print(f)
        if f[:3] == 'cat':
            labels_list.append(np.array([1., 0.]))
        else:
            labels_list.append(np.array([0., 1.]))
        imageDirs_list.append(os.path.join(file_path, f))
    # print(imageDirs_list[:5])
    imageDirs_list = np.array(imageDirs_list)
    labels_list = np.array(labels_list)
    m = imageDirs_list.shape[0]
    order = np.random.permutation(m)
    imageDirs_list = imageDirs_list[order]
    labels_list = labels_list[order]
    d = int(m*train_test)
    train_Img, test_Img = np.split(imageDirs_list, [d,])
    train_label, test_label = np.split(labels_list, [d,])
    return train_Img, train_label, test_Img, test_label

image_dir = r'E:\bawei\DeepLearning_1\深度一\tensorflow补充\catdog_data\data\train'
train_Img, train_label, test_Img, test_label = get_filePath(image_dir)
print(train_Img[1])