# -*- coding: utf-8 -*-
# @Time    : 2019/10/18 0018 8:38
# @Author  : 
# @FileName: test11.py
# @Software: PyCharm
import tensorflow as tf

a = tf.constant([[[[1,3,5,7],
                 [8,6,4,2],
                 [4,2,8,6],
                 [1,3,5,7]],
                 [[2,4,6,8],
                  [7,5,3,1],
                  [3,1,7,5],
                  [2,4,6,8]]]])
a = tf.reshape(a, [-1, 4,4,2])
pooling = tf.nn.max_pool(a, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
with tf.Session() as sess:
    print("image:")
    image = sess.run(a)
    print(image.shape)
    print(image)
    print("reslut:")
    result = sess.run(pooling)
    print (result.shape)
    print (result)
