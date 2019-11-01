# -*- coding: utf-8 -*-
# @Time    : 2019/9/26 0026 21:04
# @Author  : 
# @FileName: practice01.py
# @Software: PyCharm
import tensorflow as tf

input1 = tf.constant([1.0,2.0,3.0],name='input1')
input2 = tf.Variable(tf.random_uniform([3]), name='input2')
add = tf.add_n([input1,input2], name='addOP')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("./logs/tensorboard1",sess.graph)
    print(sess.run(add))
writer.close()

# tensorboard --logdir=logs/tensorboard1 --host=127.0.0.1
# 打开google浏览器运行：http://127.0.0.1:6006

