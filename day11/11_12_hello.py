# -*- coding: utf-8 -*-
# @Time    : 2019/10/21 0021 20:04
# @Author  : 
# @FileName: 11_12_hello.py
# @Software: PyCharm

import tensorflow as tf
import numpy as np
from tensorflow.python.ops.rnn import dynamic_rnn


# vocabulary
# 使用tensorflow框架，利用循环神经网络训练字符序列，从“hihell”学习输出“ihello”。
idx2char = ['h','i','e','l','o']
x_data = [[0, 1, 0, 2, 3, 3],   # hihell
          [0, 0, 1, 3, 2, 2]]   # hhilee


y_date = [[1, 0, 2, 3, 3, 4],   #ihello
          [0, 1, 3, 2, 2, 4]]   #hileeo
# x_one_hot = np.eye(5)[x_data].reshape(2, -1, 5)
x_one_hot = tf.one_hot(x_data, 5)
# parameter
batch_size = 2
sequence_length = 6
nb_classes = 5
hidden_size = 8
learning_rate = 0.1

# XY
X = tf.placeholder(tf.float32, [None, sequence_length, nb_classes])
Y = tf.placeholder(tf.int32, [None, sequence_length])

# rnn_cell
cell = tf.contrib.rnn.LSTMCell(num_units=hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, last_state = dynamic_rnn(cell, X, initial_state=initial_state, dtype=tf.float32)

# full_connected
x_for_fc = tf.reshape(outputs, [-1, hidden_size])
outputs = tf.contrib.layers.fully_connected(x_for_fc, num_outputs=nb_classes, activation_fn=None)

# loss
outputs = tf.reshape(outputs, [batch_size, sequence_length, nb_classes])
weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# accuracy
predicted = tf.argmax(outputs, 2)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, np.array(y_date)), tf.float32))

# session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20):
        x_oh = sess.run(x_one_hot)
        l, _, acc = sess.run([loss, optimizer, accuracy], feed_dict={X:x_oh, Y:y_date})
    p1 = sess.run(predicted, feed_dict={X: x_oh})
    result_str = [idx2char[i] for i in p1[1]]
    print("p:  ", ''.join(result_str))





"""
import tensorflow as tf
import numpy as np

tf.set_random_seed(111)
# 建立字典
idx2char = ['h','i','e','l','o']
# 构造数据集
x_data = [0,1,1,2,3,3]
x_one_hot = np.eye(5)[x_data].reshape(1, -1, 5)
y_data = [[1,0,2,3,3,4]]
# parameter
batch_size = 1
sequence_length = 6
input_dim = 5
hidden_size = 8
num_classes = 5
learning_rate = 0.1

X = tf.placeholder(tf.float32, [None, sequence_length, input_dim])  # (？， 6， 5）
Y = tf.placeholder(tf.int32, [None, sequence_length])  # （？， 6）


cell = tf.contrib.rnn.LSTMCell(num_units=hidden_size, state_is_tuple=True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, X, initial_state=initial_state, dtype=tf.float32)
X_for_fc = tf.reshape(outputs, [-1, hidden_size])  # (6, 8)
outputs = tf.contrib.layers.fully_connected(inputs=X_for_fc,
                                            num_outputs=num_classes, activation_fn=None) # (6, 5)

outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])
weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


prediction = tf.argmax(outputs, 2)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, np.array(y_data)), tf.float32))



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(50):
        l, _, acc = sess.run([loss, train, accuracy], feed_dict={X:x_one_hot, Y:y_data})
        result = sess.run(prediction, feed_dict={X:x_one_hot})
        print(i, "loss:", l, " prediction:", result, "trueY:", y_data, acc)
        # if acc >= 1.0:
        #     break
            # 用新数据测试
        t_data = [0, 2, 3, 3, 0, 1]  # hellhi
        result = sess.run(prediction, feed_dict={X: np.eye(5)[t_data].reshape(1, -1, 5)})
        print(result)
        result_str = [idx2char[c] for c in np.squeeze(result)]
        print("\tPrediction str: ", result_str, ''.join(result_str))
        
        

"""

