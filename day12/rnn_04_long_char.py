# -*- coding: utf-8 -*-
# @Time    : 2019/10/24 0024 10:07
# @Author  : 
# @FileName: rnn_04_long_char.py
# @Software: PyCharm


"""

from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.python.ops.rnn import dynamic_rnn

tf.set_random_(777)  # reproducibility

sentence = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")

char_set = list(set(sentence)) #print(len(char_set))  #25
char_dic = {w: i for i, w in enumerate(char_set)}
hidden_size = 50  # len(char_set)    # 25

sequence_length = 10  # Any arbitrary number
data_dim = len(char_set)  # 25
num_classes = len(char_set)    # 25
learning_rate = 0.1

# 构造数据集
dataX = []
dataY = []
for i in range(0, len(sentence) - sequence_length):
    x_str = sentence[i:i + sequence_length]
    y_str = sentence[i + 1: i + sequence_length + 1]
    # print(i, x_str, '->', y_str)
    x = [char_dic[c] for c in x_str]  # x str to index 字符转数字
    y = [char_dic[c] for c in y_str]  # y str to index
    dataX.append(x)
    dataY.append(y)

batch_size = len(dataX) #  170
print(batch_size)
X = tf.placeholder(tf.int32, [None, sequence_length])
Y = tf.placeholder(tf.int32, [None, sequence_length])

X_one_hot = tf.one_hot(X, num_classes) #独热编码 #print(X_one_hot) (?, 10, 25)

# 建一个有隐藏单元的LSTM，Make a lstm cell with hidden_size (each unit output vector size)
def cell():
    # cell = rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
    # cell = rnn.GRUCell(hidden_size)
    cell = rnn.LSTMCell(hidden_size, state_is_tuple=True)
    return cell
multi_cells = rnn.MultiRNNCell([cell() for _ in range(2)], state_is_tuple=True)

# outputs:展开隐藏层 unfolding size x hidden size, state = hidden size
outputs, _states = dynamic_rnn(multi_cells, X_one_hot, dtype=tf.float32)
# 全连接层FC layer
X_for_fc = tf.reshape(outputs, [-1, hidden_size])
outputs = tf.contrib.layers.fully_connected(X_for_fc, num_classes, activation_fn=None)
# print(outputs.shape) #(?,25)
# 改变维度准备计算序列损失reshape out for sequence_loss
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes]) #(170, 10, 25)
weights = tf.ones([batch_size, sequence_length])# 所有的权重都是1 All weights are 1 (equal weights)
# 计算损失值
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
mean_loss = tf.reduce_mean(sequence_loss)
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(mean_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(300): #(500)
    _, lossval, results = sess.run(
        [train_op, mean_loss, outputs], feed_dict={X: dataX, Y: dataY})
    #print(results.shape)  (170,10,25)
    # if i == 49:
    #     for j, result in enumerate(results):  #j:[0,170)   result:(10,25)
    #         index = np.argmax(result, axis=1)
            # print(i, j, ''.join([char_set[t] for t in index]), l)
    results = sess.run(outputs, feed_dict={X: dataX})  #(170,10,25)
    for j, result in enumerate(results):
        index = np.argmax(result, axis=1)
        if j is 0:  #第一个结果10个字符组成一个句子 print all for the first result to make a sentence
            ret =''.join([char_set[t] for t in index])
        else: #其它取最后一个字符
            ret = ret + char_set[index[-1]]
    print(i, lossval, ret)
    if ret == sentence[1:]:
        break

# #输出每个结果的最后一个字符检测效果 Let's print the last char of each result to check it works
# results = sess.run(outputs, feed_dict={X: dataX})  #(170,10,25)
# for j, result in enumerate(results):
#     index = np.argmax(result, axis=1)
#     if j is 0:  #第一个结果10个字符组成一个句子 print all for the first result to make a sentence
#         # print(''.join([char_set[t] for t in index]), end='')
#         ret =''.join([char_set[t] for t in index])
#     else: #其它取最后一个字符
#         # print(char_set[index[-1]], end='')
#         ret = ret + char_set[index[-1]]

"""


import tensorflow as tf
import numpy as np
from tensorflow.python.ops.rnn import dynamic_rnn

sentence = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")


idx2char = list(set(sentence))
idx2char.sort()
char_dict = {w:i for i, w in enumerate(idx2char)}
# print(char_dict)
# sampleId = [char_dict[c] for c in sentence]
# print(len(sampleId))
x_data = []
y_data = []
for c in range(0, len(sentence) - 10):
    x_str = sentence[c:c+10]
    y_str = sentence[c+1:c+11]
    x_data.append([char_dict[w] for w in x_str])
    y_data.append([char_dict[w] for w in y_str])



batch_size = len(x_data)
sequence_length = len(x_data[0])
hidden_size = 100
nb_classes = len(char_dict)
leaning_rate = 0.1
# XY
X = tf.placeholder(tf.int32, [None, sequence_length])
Y = tf.placeholder(tf.int32, [None, sequence_length])
X_one_hot = tf.one_hot(X, nb_classes)
# LSTM
cell = tf.contrib.rnn.LSTMCell(num_units=hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, last_state = dynamic_rnn(cell, X_one_hot, dtype=tf.float32, initial_state=initial_state)


# full_connected
x_for_fc = tf.reshape(outputs, [-1, hidden_size])
outputs = tf.contrib.layers.fully_connected(x_for_fc, nb_classes, activation_fn=None)

# loss
outputs = tf.reshape(outputs, [batch_size, sequence_length, nb_classes])
weights = tf.ones([batch_size, sequence_length])
loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)

# accuracy
predicted = tf.argmax(outputs, 2)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, np.array(y_data)), tf.float32))

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=leaning_rate).minimize(loss)
# session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        l, _, acc = sess.run([loss, optimizer, accuracy], feed_dict={X:x_data, Y:y_data})
        # print(l, acc)
        results = sess.run(outputs, feed_dict={X: x_data})  # (170,10,25)
        for j, result in enumerate(results):
            index = np.argmax(result, axis=1)
            if j is 0:  # 第一个结果10个字符组成一个句子 print all for the first result to make a sentence
                ret = ''.join([idx2char[t] for t in index])
            else:  # 其它取最后一个字符
                ret = ret + idx2char[index[-1]]
        print(i, l, ret)



'''



0 167 tttttttttt 3.23111
0 168 tttttttttt 3.23111
0 169 tttttttttt 3.23111
…
499 167  of the se 0.229616
499 168 tf the sea 0.229616
499 169   the sea. 0.229616

g you want to build a ship, don't drum up people together to collect wood and don't assign them tasks and work, but rather teach them to long for the endless immensity of the sea.

'''
