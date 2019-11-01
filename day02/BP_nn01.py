# # -*- coding: utf-8 -*-
# # @Time    : 2019/9/28 0028 8:49
# # @Author  :
# # @FileName: BP_nn01.py
# # @Software: PyCharm


# BP
"""
import tensorflow as tf
import matplotlib.pylab as plt

tf.set_random_seed(111)
# data
x_data = [[.05, 0.10]]
y_data = [[.01, .99]]

# X,Y,W,b
X = tf.placeholder(tf.float32, [None, 2])
Y = tf.placeholder(tf.float32, [None, 2])
w1 = tf.Variable(tf.random_normal([2, 2]), name='w1')
w2 = tf.Variable(tf.random_normal([2, 2]), name='w2')
b1 = tf.Variable(tf.random_normal([2, 2]), name='b1')
b2 = tf.Variable(tf.random_normal([2, 2]), name='b2')

# hypothesis
hypothesis1 = tf.matmul(X, w1)
a1 = tf.sigmoid(hypothesis1)
hypothesis2 = tf.matmul(a1, w2)
a2 = tf.sigmoid(hypothesis2)

# cost
cost = -tf.reduce_mean(Y*tf.log(a2) + (1-Y)*tf.log(1-a2))
cost_history = []
# BP
dz2 = a2 - Y
dw2 = tf.matmul(tf.transpose(a1), dz2)/tf.cast(tf.shape(a1)[0], tf.float32)
db2 = tf.reduce_mean(dz2)

dz1 = tf.matmul(dz2, tf.transpose(w2)) * (a1*(1-a1))
dw1 = tf.matmul(tf.transpose(X), dz1)/tf.cast(tf.shape(X)[0], tf.float32)
db1 = tf.reduce_mean(dz1)

# update
learning_rate = 0.1
update = [
    tf.assign(w1, w1-learning_rate*dw1),
    tf.assign(b1, b1-learning_rate*db1),
    tf.assign(w2, w2-learning_rate*dw2),
    tf.assign(b2, b2-learning_rate*db2)
]
# accuracy
predicted = tf.cast(a2>0.5, tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), tf.float32))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(3001):
        _, cost_val = sess.run([update, cost], feed_dict={X:x_data, Y:y_data})
        a = sess.run(a2, feed_dict={X:[[0.2,0.02]]})
        print(a)
        if i%100 == 0:
            print(cost_val)
            cost_history.append(cost_val)

    plt.plot(cost_history)
    plt.show()



"""



#
import tensorflow as tf
import matplotlib.pylab as plt
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
# X, Y W, b
X = tf.placeholder(tf.float32, [None, 4])
Y = tf.placeholder(tf.float32, [None, 3])
W = tf.Variable(tf.random_normal([4, 3]), name='weight')
b = tf.Variable(tf.random_uniform([3]), name='bias')

# hypothesis
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
# cost
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), axis=1))
cost_history = []
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
# accuracy
# predicted = tf.cast(hypothesis>0, tf.float32)
# accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), tf.float32))
prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(y_data, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2001):
        _, cost_val, acc = sess.run([train, cost, accuracy], feed_dict={X:x_data, Y:y_data})
        if i%100 == 0:
            print(cost_val, acc)
            cost_history.append(cost_val)
    plt.plot(cost_history)
    plt.show()





"""
import tensorflow as tf
import matplotlib.pyplot as plt
tf.set_random_seed(777)  #设置随机种子
learning_rate = 0.1
# 定义数据集
x_data = [[.05, 0.10]]
y_data = [[.01, .99]]
#定义占位符
X = tf.placeholder(tf.float32, [None, 2])
Y = tf.placeholder(tf.float32, [None, 2])
#模型和前向传播
W1 = tf.Variable([[.15, .25],
                  [.20, .30]])
b1 = tf.Variable([[.35, .35]])
a1 = tf.sigmoid(tf.matmul(X, W1) + b1)

W2= tf.Variable([[.40, .50],
                 [.45, .55]])
b2 = tf.Variable([[.60, .60]])
a2 = tf.sigmoid(tf.matmul(a1, W2) + b2)
# 代价或损失函数
cost = -tf.reduce_mean(Y * tf.log(a2) + (1 - Y) * tf.log(1 - a2))
cost_history = [] # 损失值列表
#BP反向传播
#第2层
dz2 = a2 - Y
dW2 = tf.matmul(tf.transpose(a1), dz2) / tf.cast(tf.shape(a1)[0], dtype=tf.float32)
db2 = tf.reduce_mean(dz2)
#第1层
da1 = tf.matmul(dz2, tf.transpose(W2))
dz1 = da1 * a1 * (1 - a1)
dW1 = tf.matmul(tf.transpose(X), dz1) / tf.cast(tf.shape(X)[0], dtype=tf.float32)
db1 = tf.reduce_mean(dz1, axis=0)
# 参数更新
update = [
  tf.assign(W2, W2 - learning_rate * dW2),
  tf.assign(b2, b2 - learning_rate * db2),
  tf.assign(W1, W1 - learning_rate * dW1),
  tf.assign(b1, b1 - learning_rate * db1)
]
# 创建会话
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) #全局变量初始化
    # 迭代训练
    for step in range(3001):
        _, cost_val = sess.run([update, cost], feed_dict={X: x_data, Y: y_data})
        if step % 100 == 0:# 显示损失值收敛情况
            print(step, "Cost: ", cost_val)
            cost_history.append(cost_val)
    # 画学习曲线
    plt.plot(cost_history[1: len(cost_history)])
    plt.show()

"""