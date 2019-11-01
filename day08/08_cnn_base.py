# -*- coding: utf-8 -*-
# @Time    : 2019/10/14 0014 19:32
# @Author  : 
# @FileName: 08_cnn_base.py
# @Software: PyCharm
import tensorflow as tf

input = tf.Variable(tf.random_normal([10, 32 , 32, 3]))

filter1 = tf.Variable(tf.random_normal([3, 3, 3, 32]))
filter2 = tf.Variable(tf.random_normal([3, 3, 3, 64]))
filter3 = tf.Variable(tf.random_normal([5, 5, 3, 4]))
filter4 = tf.Variable(tf.random_normal([5, 5, 3, 4]))
filter5 = tf.Variable(tf.random_normal([3, 3, 3, 4]))
filter6 = tf.Variable(tf.random_normal([3, 3, 3, 4]))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

result = tf.nn.conv2d(input, filter1, strides=[1, 1, 1, 1], padding='VALID')
print(input.shape, filter1.shape, 'strides=1 valid(p=0)==>', result.shape,'32-3+2p/1 + 1')
result = tf.nn.max_pool(result, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print('pool 2', result.shape)


result = tf.nn.conv2d(input, filter2, strides=[1,1,1,1], padding='SAME')
print(input.shape, filter2.shape, 'strides=1 valid(p=1)==>', result.shape, '(32-3+2p)/1 + 1')
result = tf.nn.max_pool(result, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print('pool 2', result.shape)

result = tf.nn.conv2d(input, filter3, strides=[1,2,2,1], padding='VALID')
print(input.shape, filter3.shape, 'strides=2 valid(p=0)==>', result.shape, '(32-5+2p)/2 + 1')
result = tf.nn.max_pool(result, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print('pool 2', result.shape)


result = tf.nn.conv2d(input, filter4, strides=[1,2,2,1], padding='SAME')
print(input.shape, filter4.shape, 'strides=2 valid(p=2)==>', result.shape, '(32-5+2p)/2 + 1')
result = tf.nn.max_pool(result, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
print('pool 2', result.shape)

result = tf.nn.conv2d(input, filter5, strides=[1,3,3,1], padding='VALID')
print(input.shape, filter5.shape, 'strides=1 valid(p=0)==>', result.shape, '(32-5+2p)/3 + 1')
result = tf.nn.max_pool(result, ksize=[1,3,3,1], strides=[1,3,3,1], padding='SAME')
print('pool 3', result.shape)

result = tf.nn.conv2d(input, filter5, strides=[1,3,3,1], padding='SAME')
print(input.shape, filter5.shape, 'strides=3 valid(p=2)==>', result.shape, '(32-5+2p)/3 + 1')
result = tf.nn.max_pool(result, ksize=[1,3,3,1], strides=[1,3,3,1], padding='SAME')
print('pool 3', result.shape)


result = tf.nn.conv2d(input, filter6, strides=[1,3,3,1], padding='SAME')
print(input.shape, filter6.shape, 'strides=3 valid(p=0)==>', result.shape, '(32-5+2p)/3 + 1')
result = tf.nn.max_pool(result, ksize=[1,4,4,1], strides=[1,4,4,1], padding='SAME')
print('pool 4', result.shape)

sess.close()