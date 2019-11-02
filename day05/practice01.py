# -*- coding: utf-8 -*-
# @Time    : 2019/10/10 0010 15:12
# @Author  : 
# @FileName: practice01.py
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
w = tf.Variable(tf.random_normal([4, 3]),name='w')
b = tf.Variable(tf.random_normal([3]),name='b')
# hypothesis
hypothesis = tf.nn.softmax(tf.matmul(X, w) + b)
# cost
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), axis=1))
# train
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
# accuracy
prediction = tf.argmax(hypothesis, 1)
correct_pred = tf.equal(prediction, tf.argmax(y_data, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# 收集， 合并
tf.summary.scalar('cost', cost)
tf.summary.histogram('w', w)
merge = tf.summary.merge_all()
# 保存
saver = tf.train.Saver(max_to_keep=5)


# session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    fileWriter = tf.summary.FileWriter('./log', graph=sess.graph)
    # for i in range(1001):
    #     cost_val, _, acc, summary = sess.run([cost, train, accuracy, merge], feed_dict={X:x_data, Y:y_data})
    #     fileWriter.add_summary(summary, i)
    #     if i%100 == 0:
    #         print(cost_val, acc)
    #         saver.save(sess, './saved/m1')  # 具体到文件名
    saver.restore(sess, './saved/m1')
    cost_val, _, acc, summary = sess.run([cost, train, accuracy, merge], feed_dict={X: x_data, Y: y_data})
    print(cost_val, acc)
    fileWriter.close()