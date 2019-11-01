# -*- coding: utf-8 -*-
# @Time    : 2019/10/23 0023 16:34
# @Author  : 
# @FileName: 11_12_char.py
# @Software: PyCharm
# 循环神经网络，短句训练
import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn
import numpy as np
sample = " if you want you like"
idx2char = list(set(sample))
idx2char.sort()
char_dict = {w:i for i, w in enumerate(idx2char)}
sampleId = [char_dict[c] for c in sample]

# data
x_data = [sampleId[:-1]]
y_data = [sampleId[1:]]

# parameter
sequence_length = len(x_data[0])
batch_size = 1
nb_classes = len(char_dict)
hidden_size = 10
learning_rate = 0.2

#XY
X = tf.placeholder(tf.int32, [None, sequence_length])
Y = tf.placeholder(tf.int32, [None, sequence_length])
X_one_hot = tf.one_hot(X, nb_classes)
# LSTM
cell1 = tf.contrib.rnn.LSTMCell(num_units=hidden_size, state_is_tuple=True)
cell2 = tf.contrib.rnn.LSTMCell(num_units=hidden_size, state_is_tuple=True)
# initial_state = cell.zero_state(batch_size, tf.float32)
mul_cell = tf.contrib.rnn.MultiRNNCell([cell1, cell2], state_is_tuple=True)
outputs, last_state = dynamic_rnn(mul_cell, X_one_hot,dtype=tf.float32)
# fully_connected
x_for_fc = tf.reshape(outputs, [-1, hidden_size])
outputs = tf.contrib.layers.fully_connected(x_for_fc, num_outputs=nb_classes, activation_fn=None)
# loss
outputs = tf.reshape(outputs, [batch_size, sequence_length, nb_classes])
weights = tf.ones([batch_size, sequence_length])
loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights)
# accuracy
predicted = tf.argmax(outputs, 2)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, np.array(y_data)), tf.float32))
# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
# session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(50):
        l, _, acc = sess.run([loss, optimizer, accuracy], feed_dict={X:x_data, Y:y_data})
        print(l, acc)
    print("".join([idx2char[c] for c in sess.run(predicted, feed_dict={X:x_data})[0]]))

'''


import os
from tensorflow.python.ops.rnn import dynamic_rnn
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.set_random_seed(777)

# 构造数据集
sample = " if you want you like"
idx2char = list(set(sample))  # 去重放列表里，set无序13
idx2char.sort()
print(idx2char)  #[' ', 'a', 'e', 'f', 'i', 'k', 'l', 'n', 'o', 't', 'u', 'w', 'y']
char2idx = {c: i for i, c in enumerate(idx2char)}  # 转为字典  把字母作为键 它的索引作为值
print(char2idx) # {' ': 0, 'a': 1, 'e': 2, 'f': 3, 'i': 4, 'k': 5, 'l': 6, 'n': 7, 'o': 8, 't': 9, 'u': 10, 'w': 11, 'y': 12}
sample_idx = [char2idx[c] for c in sample]  # 在字典里取出对应值
print(sample_idx) # [0, 4, 3, 0, 12, 8, 10, 0, 11, 1, 7, 9, 0, 12, 8, 10, 0, 6, 4, 5, 2]
x_data = [sample_idx[:-1]]  # 输入去掉最后一个
y_data = [sample_idx[1:]]  # 输出去掉第一个

# 设置构建RNN所需要的参数
dic_size = len(char2idx) #字典长度13
rnn_hidden_size = len(char2idx)*2  # 隐藏层单元个数26 cell中神经元的个数，不一定要和序列数相等
batch_size = 1  #  批大小
sequence_length = len(sample) - 1  # 序列长度（时间步数）20
num_classes = len(char2idx) # 最终输出大小13（RNN或softmax等）

# 定义占位符并且进行独热编码的转化
X = tf.placeholder(tf.int32, [None, sequence_length])  # X data(?, 20)
Y = tf.placeholder(tf.int32, [None, sequence_length])  # Y label(?, 20)
X_one_hot = tf.one_hot(X, num_classes) # 变为独热编码: 1 -> 0 1 0 0 0 0 0 0 0 0 0 0 0
# print(X_one_hot.shape)  #(?, 20, 13)

# 构建RNN
# cell = tf.contrib.rnn.BasicLSTMCell(num_units=rnn_hidden_size, state_is_tuple=True) #必须写它规定了状态信息的格式
# cell = tf.contrib.rnn.LSTMCell(num_units=rnn_hidden_size, state_is_tuple=True) #必须写它规定了状态信息的格式
cell = tf.contrib.rnn.LSTMCell(num_units=rnn_hidden_size, state_is_tuple=True)
# cell = tf.contrib.rnn.GRUCell(num_units=rnn_hidden_size)
initial_state = cell.zero_state(batch_size, tf.float32) #RNN的初始化状态
outputs, _states = dynamic_rnn(cell, X_one_hot, initial_state=initial_state, dtype=tf.float32)
#print(outputs.shape) # (1,20,26)
outputs = tf.reshape(outputs,[-1,rnn_hidden_size]) # 全连接前需要变为二维数据 (20,26)
#加一层全连接，相当于加一层深度，使预测更准确
outputs = contrib.layers.fully_connected(inputs=outputs, num_outputs=num_classes, activation_fn=None)
#print(outputs.shape) #经过一层全连接 变为(1*20,13) [batch_size*sequence_length,num_classes]

outputs = tf.reshape(outputs,[batch_size,sequence_length,num_classes])#变为3维(1,20,13)
weights = tf.ones([batch_size, sequence_length]) #weight为t时与t+1时之间的权重
#计算序列损失
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=Y, weights=weights) #三维数据
loss = tf.reduce_mean(sequence_loss)

train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

# 预测值
prediction = tf.argmax(outputs, axis=2) #最后的outputs是三维的 所以axis=2
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        sl, l, _ = sess.run([sequence_loss, loss, train], feed_dict={X: x_data, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_data})
        #用字典输出字符 print char using dic
        result_str = [idx2char[c] for c in np.squeeze(result)]
        print(i, sl, "损失:", l, "预测:[", ''.join(result_str),']')
        if sample[1:]==''.join(result_str):
            break
0 2.563072 损失: 2.563072 预测:[ o                    ]
1 2.3728235 损失: 2.3728235 预测:[ u  uuuiiiiiiiiiiiiii ]
2 2.4159513 损失: 2.4159513 预测:[ yyyfuuuuunuyyuuuuuff ]
3 2.2046945 损失: 2.2046945 预测:[ yyyyuu   n   uu    e ]
4 1.8944768 损失: 1.8944768 预测:[ yyyyou   nt you   ke ]
5 1.5556475 损失: 1.5556475 预测:[ yfyyou  nntyyou   ee ]
6 1.210151 损失: 1.210151 预测:[ yfyyou wantyyou wike ]
7 0.8979864 损失: 0.8979864 预测:[ yfyyou want you wike ]
8 0.6207568 损失: 0.6207568 预测:[ yf you lant you like ]
9 0.42925024 损失: 0.42925024 预测:[ yf you want you like ]
10 0.28728548 损失: 0.28728548 预测:[ yf you want you like ]
11 0.18484735 损失: 0.18484735 预测:[ if you want you like ]
'''
