# -*- coding: utf-8 -*-
# @Time    : 2019/9/27 0027 10:29
# @Author  : 
# @FileName: tensorFlow_01_Regression1.py
# @Software: PyCharm

"""
import tensorflow as tf
# seed
tf.set_random_seed(111)
# data
x_data = [[1.3, 2.2],
          [1.2, 2.4],
          [2.1, 1.9]]
y_data = [[6.2],
          [6.8],
          [7.4]]

# X Y W b
X = tf.placeholder(tf.float32, shape=[None, 2], name='X')
Y = tf.placeholder(tf.float32, shape=[None, 1], name='Y')

W = tf.Variable(tf.random_normal([2, 1]), dtype=tf.float32,name='weight')
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32, name='bias')
# hypothesis
hypothesis = tf.matmul(X, W) + b
# cost
cost = tf.reduce_mean(tf.square(hypothesis - Y))
# train
train = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(cost)
dw = tf.matmul(tf.transpose(X), (hypothesis - Y))
db = tf.reduce_mean((hypothesis - Y))

# update
learning_rate = 10e-5
update = [tf.assign(W, W-learning_rate*dw),
          tf.assign(b, b-learning_rate*db)]
# Session
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(2001):
    cost_val, _ = sess.run([cost, update],
                           feed_dict={X:x_data,Y:y_data})
    if i%100 == 0:
        print(cost_val)
# fileW = tf.summary.FileWriter('./logs/tensorboard1', graph=sess.graph)
# fileW.close()
sess.close()


"""


# 熟悉逻辑回归代码

"""
import tensorflow as tf
# seed
tf.set_random_seed(111)
# data
x_data = [[1, 2],
          [2, 3],
          [3, 1],
          [4, 3],
          [5, 3],
          [6, 2]]
y_data = [[0],
          [0],
          [0],
          [1],
          [1],
          [1]]
# X, Y, W, b

X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# hypothesis
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)
# cost
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))
# gradientDescent
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
# accuracy
predicted = tf.cast(hypothesis>0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(5001):
    cost_val, acc, _ = sess.run([cost, accuracy, train],
                                feed_dict={X:x_data,
                                           Y:y_data})
    if i%500 == 0:
        print(i, cost_val, acc)
print("*"*100)
#准确率
h, c, a = sess.run([hypothesis, predicted, accuracy],
                   feed_dict={X: x_data, Y: y_data})

print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)

#测试
h1, p1 = sess.run([hypothesis, predicted], feed_dict={X: [[1, 1]]})
print(h1, p1)
h2, p2 = sess.run([hypothesis, predicted], feed_dict={X: [[4,1], [3,100]]})
print('\n', h2, '\n', p2)
# 以下代码临时测试用
while True:
    str = input()
    try:
        if str == 'q':
            break
        test = list(map(float,str.split(',')))
        h1, p1 = sess.run([hypothesis, predicted], feed_dict={X: [test]})
        print(h1, p1)
    except:
        continue


sess.close()



"""



# 熟悉多变量线性回归代码4-2.py

"""
import tensorflow as tf
# seed
tf.set_random_seed(111)
# data
x_data = [[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]]

y_data = [[152.],
          [185.],
          [180.],
          [196.],
          [142.]]
# X, Y, weight, bias
X = tf.placeholder(dtype=tf.float32, shape=[None, 3])
Y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3 ,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# model
hypothesis = tf.matmul(X,W) + b

# cost
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# gradientDescent
train = tf.train.GradientDescentOptimizer(learning_rate=0.000001).minimize(cost)

# session
sess = tf.Session()
# global_variables_initializer
sess.run(tf.global_variables_initializer())
for i in range(2000):
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train],
                                         feed_dict={X:x_data,
                                                    Y:y_data})
    # if i%100 == 0:
    #     print(cost_val, '\n', W_val, '\n', b_val)

# predict
print(sess.run(hypothesis, feed_dict={X: [[5, 6, 7]]}))

# close
sess.close()




"""


# 初步练习单变量线性回归代码


"""
"""
import tensorflow as tf
tf.set_random_seed(111)
x_data = [1,2,3]
y_data = [1.1, 2.11, 3.09]
X = tf.placeholder('float', shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')


hypothesis = X * W + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
# print(sess.run(W))
for i in range(2001):
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train],
                                         feed_dict={X: x_data,
                                                    Y: y_data})
    if i%100 == 0:
        print(i, cost_val, W_val, b_val)
fileW = tf.summary.FileWriter('./logs/tensorBoard', graph=sess.graph)
fileW.close()
sess.close()
