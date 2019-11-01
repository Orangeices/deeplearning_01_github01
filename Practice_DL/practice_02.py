# -*- coding: utf-8 -*-
# @Time    : 2019/9/27 0027 9:44
# @Author  : 
# @FileName: practice_02.py
# @Software: PyCharm

# fileWriter = tf.summary.FileWriter('./logs/tensorboard1', graph=sess.graph)
# fileWriter.close()

# softmax多分类代码第1种方式6-1.py
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
# X Y W b"
X = tf.placeholder(tf.float32, [None, 4])
Y = tf.placeholder(tf.float32, [None, 3])

W = tf.Variable(tf.random_normal([4, 3]), dtype=tf.float32,name='weight')
b = tf.Variable(tf.random_normal([3]), dtype=tf.float32,name='bias')

# hypothesis
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
# cost
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), axis=1))
# gradDescent
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
# accuracy
prediction = tf.argmax(hypothesis, 1)
correct_predicted = tf.equal(prediction, tf.argmax(y_data, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predicted, tf.float32))
# Session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2001):
        cost_val, _, acc = sess.run([cost, train, accuracy], feed_dict={X:x_data,Y:y_data})
        if i%500 == 0:
            print(cost_val, "accuracy:", acc)
    # while True:
    #     Str = input()
    #     try:
    #         if Str == 'exit':
    #             break
    #         test = list(map(float, Str.split(',')))
    #         h1, p1 = sess.run([hypothesis, prediction], feed_dict={X:[test]})
    #         print(h1, '\n', p1)
    #     except:
    #         continue
    while True:
        str = input()
        try:
            if str == 'exit':
                break
            test = list(map(float, str.split(',')))
            print(test)
            h1, p1 = sess.run([hypothesis, prediction], feed_dict={X: [test]})
            print(h1, p1)
        except:
            continue
