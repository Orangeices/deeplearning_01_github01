# -*- coding: utf-8 -*-
# @Time    : 2019/10/25 0025 8:36
# @Author  : 
# @FileName: test18.py
# @Software: PyCharm
# 使用tensorflow框架，利用循环神经网络进行短句训练“ china is the best”。
import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn
import numpy as np
sample = ' china is the best'
# (一)	建立字典（8分）
idx2char = list(set(sample))
idx2char.sort()
char_dict = {w:i for i , w in enumerate(idx2char)}
# print(char_dict)
sampleId = [char_dict[c] for c in sample]
print(sampleId)
# (二)	构造数据集x_data,y_data（8分）
x_data = [sampleId[:-1]]
y_data = [sampleId[1:]]
# (三)	设置参数（8分）
batch_size = 1
sequence_length = len(x_data[0])
nb_classes = len(char_dict)
hidden_size = 10
# (四)	定义占位符（8分）
X = tf.placeholder(tf.int32, [None, sequence_length])
Y = tf.placeholder(tf.int32, [None, sequence_length])
X_one_hot = tf.one_hot(X, nb_classes)
# (五)	定义LSTM单元（4分）、堆叠多层RNN单元（4分）、调用动态RNN函数（4分）
cell1 = tf.contrib.rnn.LSTMCell(num_units=hidden_size, state_is_tuple=True)
cell2 = tf.contrib.rnn.LSTMCell(num_units=hidden_size, state_is_tuple=True)
mult_cells = tf.contrib.rnn.MultiRNNCell([cell1, cell2], state_is_tuple=True)
outputs, last_state = dynamic_rnn(mult_cells, X_one_hot, dtype=tf.float32)
# (六)	定义全连接层（8分）
x_for_fc = tf.reshape(outputs, [-1, hidden_size])
outputs = tf.contrib.layers.fully_connected(x_for_fc, num_outputs=hidden_size, activation_fn=None)

# (七)	计算序列损失（8分）
outputs = tf.reshape(outputs, [batch_size, sequence_length, nb_classes])
weights = tf.ones([batch_size, sequence_length])
loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
# (八)	定义准确率计算模型（8分）
predicted = tf.argmax(outputs, 2)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, np.array(y_data)), tf.float32))
#
optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

# (九)	开始训练迭代100次（8分）
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        l, _, acc = sess.run([loss, optimizer, accuracy], feed_dict={X:x_data, Y:y_data})
# (十)	输出损失值、准确率（8分）
        print(l,"  ", acc)
# (十一)	预测结果查字典后输出字符串（8分）
    print("".join([idx2char[c] for c in sess.run(predicted, feed_dict={X:x_data})[0]]))
# (十二)	用一个新的数据“the best is china”进行测试，以字符串格式输出预测的结果（8分）
    t_data = 'the best is china'
    print([[char_dict[c] for c in t_data]][0])
    print("".join([idx2char[c] for c in sess.run(predicted,
                                                 feed_dict={X: [[char_dict[c] for c in t_data]]})[0]]))
