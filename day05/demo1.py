# -*- coding: utf-8 -*-
# @Time    : 2019/10/10 0010 10:09
# @Author  : 
# @FileName: demo1.py
# @Software: PyCharm
import tensorflow as tf
tf.set_random_seed(111)
# data

x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

# xywb
X = tf.placeholder(tf.float32, [None, 4])
Y = tf.placeholder(tf.float32, [None, 3])

with tf.Session() as sess:
    load_graph = tf.train.import_meta_graph('./saved/m1.meta')
    load_graph.restore(sess, './saved/m1')
    graph = tf.get_default_graph()
    w = graph.get_tensor_by_name('w:0')
    b = graph.get_tensor_by_name('b:0')
    hypothesis = tf.nn.softmax(tf.matmul(X, w) + b)
    # hypothesis = graph.get_tensor_by_name('h:0')
    # hypothesis = tf.nn.softmax(tf.matmul(X, w) + b, name='h')
    prediction = graph.get_tensor_by_name('prediction:0')
    correct_pred = graph.get_tensor_by_name('correct_pred:0')
    accuracy = graph.get_tensor_by_name('accuracy:0')
    prediction = tf.argmax(hypothesis, 1, name='prediction')
    correct_pred = tf.equal(prediction, tf.argmax(y_data, 1), name='correct_pred')
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
    # print(accuracy)
    h = sess.run([hypothesis], feed_dict={X: x_data, Y: y_data})
    print(h)

    # cost_val, _, acc, summary = sess.run([cost, train, accuracy, merge], feed_dict={X: x_data, Y: y_data})
    # print(cost_val, acc)