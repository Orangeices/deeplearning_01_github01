# -*- coding: utf-8 -*-
# @Time    : 2019/10/24 0024 8:35
# @Author  : 
# @FileName: test17.py
# @Software: PyCharm
# 使用tensorflow框架，利用循环神经网络训练字符序列，从“hihell”学习输出“ihello”。
import tensorflow as tf
import numpy as np
from tensorflow.python.ops.rnn import dynamic_rnn
# (一)	建立字典，有5个字符：h,i,e,l,o（8分）
d = ['h','i','e', 'l', 'o']
# (二)	构造数据集x_data,y_data（8分）
x_data = [0,1,0,2,3,3]  # hihell
y_data = [[1,0,2,3,3,4]]  # ihello
x_one_hot = np.eye(5)[x_data].reshape(1, -1, 5)
# (三)	设置参数（8分）
sequence_length = 6
hidden_size = 8
batch_size = 1
learning_rate = 0.1
nb_classes = 5
# (四)	定义占位符（8分）
X = tf.placeholder(tf.float32, [None, sequence_length, nb_classes])
Y = tf.placeholder(tf.int32, [None, sequence_length])
# (五)	定义LSTM单元（4分）、设置初始状态（4分）、调用动态RNN函数（4分）
cell = tf.contrib.rnn.LSTMCell(num_units=hidden_size, state_is_tuple=True, dtype=tf.float32)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, last_state = dynamic_rnn(cell, X, initial_state=initial_state, dtype=tf.float32)
# (六)	定义全连接层（8分）
x_for_fc = tf.reshape(outputs, [-1, hidden_size])
outputs = tf.contrib.layers.fully_connected(x_for_fc, num_outputs=nb_classes, activation_fn=None)
# (七)	计算序列损失（8分）
outputs = tf.reshape(outputs, [batch_size, sequence_length, nb_classes])
weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
# (八)	定义准确率计算模型（8分）
predicted = tf.argmax(outputs, 2)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, np.array(y_data)), tf.float32))

# (九)	开始训练迭代（8分）

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20):
        l, _, acc = sess.run([loss, optimizer, accuracy], feed_dict={X:x_one_hot, Y:y_data})
# (十)	输出损失值、准确率（8分）
        print(l, acc)
# (十一)	预测结果查字典后输出字符串（8分）
        pred = sess.run(predicted, feed_dict={X:x_one_hot})
        r1 = [d[i] for i in pred[0]]
        print("prediction:  ", "".join(r1))
        if acc>=1.0:
            break
# (十二)	准确度达到1.0后退出训练（4分）
# (十三)	用一个新的数据“hellhi”进行测试，以字符串格式输出预测的结果（4分）
    test_d = [0,2,3,3,0,1]
    result = sess.run(predicted, feed_dict={X:np.eye(5)[test_d].reshape(1, -1, 5)})
    r2 = [d[i] for i in result[0]]
    print("".join(r2))