# -*- coding: utf-8 -*-
# @Time    : 2019/10/9 0009 16:21
# @Author  : 
# @FileName: 6-1.py
# @Software: PyCharm


# softmax多分类代码第1种方式6-1.py：
# import tensorflow as tf
# # seed
# tf.set_random_seed(111)
# # data
#
# x_data = [[1, 2, 1, 1],
#           [2, 1, 3, 2],
#           [3, 1, 3, 4],
#           [4, 1, 5, 5],
#           [1, 7, 5, 5],
#           [1, 2, 5, 6],
#           [1, 6, 6, 6],
#           [1, 7, 7, 7]]
# y_data = [[0, 0, 1],
#           [0, 0, 1],
#           [0, 0, 1],
#           [0, 1, 0],
#           [0, 1, 0],
#           [0, 1, 0],
#           [1, 0, 0],
#           [1, 0, 0]]
# # XYwb
# X = tf.placeholder(tf.float32, [None, 4])
# Y = tf.placeholder(tf.float32, [None, 3])
# w =tf.Variable(tf.random_normal([4, 3]), name='weight')
# b = tf.Variable(tf.random_normal([3]), name='bias')
# # hypothesis
# hypothesis = tf.nn.softmax(tf.matmul(X, w) + b)
# # cost
# cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), axis=1))
# # train
# train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
# # accuracy
# prediction = tf.argmax(hypothesis, 1)
# correct_pred = tf.equal(prediction, tf.argmax(y_data, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# # a, accuracy = tf.metrics.accuracy(tf.argmax(y_data, 1), tf.argmax(hypothesis, 1))
# # session
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     sess.run(tf.local_variables_initializer())
#     for i in range(5001):
#         cost_val, _, acc = sess.run([cost, train, accuracy], feed_dict={X:x_data, Y:y_data})
#         if i%500 == 0:
#             print(cost_val, acc)
#     while True:
#         str = input()
#         try:
#             if str == 'q':
#                 break
#             else:
#                 test = list(map(float, str.split(',')))
#                 h1, p1 = sess.run([hypothesis, prediction], feed_dict={X:[test]})
#                 print(h1, p1)
#         except:
#             continue



# 把标签从数字转换为独热编码。 使用tf.nn.softmax_cross_entropy_with_logits_v2 函数。
"""
import tensorflow as tf
import numpy as np
tf.set_random_seed(111)
path = r'E:\bawei\DeepLearning_1\深度一\tensorflow补充'
# data
xy = np.loadtxt(path+'\data-04-zoo.csv', delimiter=',')
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

nb_classes = 7
X = tf.placeholder('float', [None, 16])
Y = tf.placeholder(tf.int32, [None, 1])
Y_one_hot = tf.one_hot(Y, nb_classes)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
w1 = tf.Variable(tf.random_normal([16, nb_classes]), name='weight1')
w2 = tf.Variable(tf.random_normal([nb_classes, nb_classes]), name='weight2')
b1 = tf.Variable(tf.random_normal([nb_classes]), name='bias1')
b2 = tf.Variable(tf.random_normal([nb_classes]), name='bias2')
# hypothesis
logits1 = tf.matmul(X, w1) + b1
hypothesis1 = logits1

logits2 = tf.matmul(hypothesis1, w2) + b2
hypothesis2 = logits2
# cost
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits2, labels=Y_one_hot))
# train
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
# accuracy
prediction = tf.argmax(hypothesis2, 1)
correct_pred = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2001):
        cost_val, _, acc = sess.run([cost, train, accuracy], feed_dict={X:x_data, Y:y_data})
        if i%100 == 0:
            print(cost_val, acc)

    while True:
        try:
            str = input()
            if str == 'q':
                break
            test = list(map(float, str.split(',')))
            h1, p1 = sess.run([hypothesis2, prediction], feed_dict={X:[test]})
        except:
            continue

"""




# import tensorflow as tf
"""
‘框架之争’
PyTorch与TensorFlow成为最后两大玩家，PyTorch占据学术界领军地位，
TensorFlow在工业界力量强大。
"""
# import numpy as np
"""
用于计算、处理多维数组的python包,大部分是用c语言编写的,
NumPy提供各种强大的数据结构(多维数组和矩阵)，以及对这些数据结构的强大运算能力。
TensorFlow基于Numpy
"""
# tf.set_random_seed(777) #设置随机种子
"""
random seed操作其实分为两种：
graph-level（图级）和op-level（操作级），
随机数生成种子是在数据流图资源上运作的，接下来让我具体介绍它们。

第一种情况：要在Session中生成不同的序列，既不设置图级别也不设置op级别种子：

a = tf.random_uniform([1])
b = tf.random_normal([1])

print( "Session 1")
with tf.Session() as sess1:
    print (sess1.run(a))  # generates 'A1'
    print (sess1.run(a))  # generates 'A2'
    print (sess1.run(b))  # generates 'B1'
    print (sess1.run(b))  # generates 'B2'
 
print( "Session 2")
with tf.Session() as sess2:
    print (sess2.run(a))  # generates 'A3'
    print (sess2.run(a))  # generates 'A4'
    print (sess2.run(b))  # generates 'B3'
    print (sess2.run(b))  # generates 'B4'
实验结果：可以明显看出，无论是在同一个Session还是在不同的Session中，生成的序列都不同。

第二种情况：要为跨Session生成相同的可重复序列，请为op设置种子：
 
a = tf.random_uniform([1], seed=1)     #op-level 随机生成种子
b = tf.random_normal([1])
 
print( "Session 1")
with tf.Session() as sess1:
    print (sess1.run(a))  # generates 'A1'
    print (sess1.run(a))  # generates 'A2'
    print (sess1.run(b))  # generates 'B1'
    print (sess1.run(b))  # generates 'B2'
 
print( "Session 2")
with tf.Session() as sess2:
  print (sess2.run(a))  # generates 'A3'
  print (sess2.run(a))  # generates 'A4'
  print (sess2.run(b))  # generates 'B3'
  print (sess2.run(b))  # generates 'B4'
实验结果：明显可以看出在op-level级随机生成种子的操作后，
同一个Session内生成不同的序列，跨Session生成相同的序列。
第三种情况：要使所有生成的随机序列在会话中可重复，就要设置图级别的种子：
 
tf.set_random_seed(1234)
a = tf.random_uniform([1])
b = tf.random_normal([1])
 
print( "Session 1")
with tf.Session() as sess1:
    print (sess1.run(a))  # generates 'A1'
    print (sess1.run(a))  # generates 'A2'
    print (sess1.run(b))  # generates 'B1'
    print (sess1.run(b))  # generates 'B2'
 
print( "Session 2")
with tf.Session() as sess2:
    print (sess2.run(a))  # generates 'A3'
    print (sess2.run(a))  # generates 'A4'
    print (sess2.run(b))  # generates 'B3'
    print (sess2.run(b))  # generates 'B4'
  
明显可以看出，跨Session生成的所有序列都是重复的，
但是在档额Session里是不同的，这就是graph-level的随机生成种子。
这tf.set_random_seed(interger)  中不同的interger没有什么不同，
只是相同的interger每次生成的序列是固定的。

"""
# #定义数据集
# path = r'E:\bawei\DeepLearning_1\深度一\tensorflow补充'
# xy = np.loadtxt(path+'\data-04-zoo.csv', delimiter=',')
"""
np.loadtxt(fname,dtype=np.float,delimiter=None,skiprows=0,usecols=None,unpack=False)
fname：文件、字符串或产生器，可以是.gz或bz2压缩文件

　　dtype：数据类型，可选，CSV的字符串以什么数据类型读入数组中，默认np.float

　　delimiter：分割字符串，默认是任何空格，*改为逗号*

　　skiprows：跳过前x行，一般跳过第一行表头

　　usecols：读取指定的列，索引，元组类型

　　unpack：如果True，读入属性将分别写入不同数组变量，False读入数据只写入一个数组变量，默认False
"""

# x_data = xy[:, 0:-1]
# y_data = xy[:, [-1]]

"""
import numpy as np

t1 = np.arange(12).reshape(3,4)
print(t1)
#取行
print(t1[2],t1[1],t1[0])
#取连续的多行
print(t1[1:])
#取不连续的多列
print(t2[:,[0,2]])
#取不连续的多行
print(t1[[0,2]])
print(t1[1,1:])
print(t1[:,1])
#取多个不相邻的点
#选出来的结果是（0，0）（2，2）
print(t1[[0,2],[0,2]])

"""
# nb_classes = 7  # 0 ~ 6
# #定义占位符
# X = tf.placeholder("float", shape=[None, 16])
# Y = tf.placeholder(tf.int32, [None, 1])  # 标签0 ~ 6
"""
1.placeholder 机制的作用:
网络的输入数据是一个矩阵，我们把多个这样的矩阵数据打包成一个很大的数据集，
如果将这个数据集当作变量或常量一下子输入到网络中，那么就需要定义很多的网络输入常量，
于是计算图上将会涌现大量的输入节点。这是不利的，这些节点的利用率很低。
placehoder 机制被设计用来解决这个问题。编程时只需要将数据通过 placeholder 传入 TensorFlow 计算图即可。

2.使用方法：
在 placeholder 定义时，这个位置上的数据类型dtype是需要指定且不可以改变的。
placeholder 中数据的维度信息shape可以根据提供的数据推导得出，所以不一定要给出；或者对于不确定的维度，填入None即可。
这里输入a和b定义为常量，这里将它们定义为一个tf.placeholder()，
在运行会话时需要通过sess.run()函数的feed_dict来提供a和b的取值。
feed_dict是一个字典dict，在字典中需要给出每个用到的placeholder的取值。
"""
# Y_one_hot = tf.one_hot(Y, nb_classes)  # 转换为独热编码格式，输出张量维度(?, 1, 7)
# '''
# [
#     [[1,0,0,0,0,0,0]]
#     [[0,1,0,0,0,0,0]]
# ]
# '''
# print(Y_one_hot.shape)
# Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes]) # 去掉1维的部分，输出张量维度(?, 7)
# '''
# [
#     [1,0,0,0,0,0,0]
#     [0,1,0,0,0,0,0]
# ]
# '''
# print(Y_one_hot.shape)
# #权重和偏置
# w1 = tf.Variable(tf.random_normal([16, nb_classes]), name='weight1')
# w2 = tf.Variable(tf.random_normal([nb_classes, nb_classes]), name='weight2')
# b1 = tf.Variable(tf.random_normal([nb_classes]), name='bias1')
# b2 = tf.Variable(tf.random_normal([nb_classes]), name='bias2')
# 预测模型  (?, 16) * (16, 10) + (1, 10) --> (1, 10) * (10, 10) + (1,10) --> (? ,10) 存在广播机制
# logits1 = tf.matmul(X, w1) + b1
# hypothesis1 = logits1
#
# logits2 = tf.matmul(hypothesis1, w2) + b2
# hypothesis2 = logits2
"""
1.tf.Variable(initial_value,
             trainable=True, 
             collections=None, 
             validate_shape=True, 
             name=None)
             
参数名称	                参数类型	                        含义
initial_value	所有可以转换为Tensor的类型	变量的初始值,一般是随机生成函数的值
trainable	            bool	            是否加入到GraphKeys.TRAINABLE_VARIABLES被迭代优化
collections	            list	            指定该图变量的类型、默认为GraphKeys.GLOBAL_VARIABLES
validate_shape	        bool	            是否进行类型和维度检查
name	                string	            变量的名称，如果没有指定则系统会自动分配一个唯一的值

2.在sess对变量运算前初始化所有变量
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    
3.管理变量的变量空间
with tf.variable_scope("one") :
    a = tf.get_variable ("a",shape=[1],initializer=tf.constant_initializer(1.0))
以上代码在名为one的变量空间内创建名字为a的变量;
因为tf.variable_scope("one")的参数默认reuse=False,
所以在one这个变量空间内不能在创建名字为a的变量；
若reuse=True则get_variable()函数会直接获取name属性相同的己经创建的变量,
获取的变量没创建过则会报错（区别于指定initializer时为创建新变量）
"""
# #预测模型
# #logits = tf.matmul(X, W) + b
# #hypothesis = logits #预测模型之一，只用于softmax_cross_entropy_with_logits
# # hypothesis = tf.nn.softmax(logits)  #预测模型之二，都可用
# #代价或损失函数
# # cost = tf.reduce_mean(-tf.reduce_sum(Y_one_hot * tf.log(hypothesis2), axis=1))
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits2, labels=Y_one_hot))
"""
1.tf.nn.softmax_cross_entropy_with_logits原理是先对logits做softmax之后，
在于label做交叉熵运算

2.tf.nn.sigmoid_cross_entropy_with
它对于输入的logits先通过sigmoid函数计算，再计算它们的交叉熵，
但是它对交叉熵的计算方式进行了优化，使得结果不至于溢出，
它适用于每个类别相互独立但互不排斥的情况：
例如一幅图可以同时包含一条狗和一只大象 output不是一个数，而是一个batch中每个样本的cost,
所以一般配合tf.reduce_mea(loss)使用
"""
# #梯度下降优化器
# train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
"""

①tf.train.GradientDescentOptimizer()
使用随机梯度下降算法，使参数沿着梯度的反方向，即总损失减小的方向移动，实现更新参数

其中，𝐽(𝜃)为损失函数，𝜃为参数，𝛼为学习率。

②tf.train.MomentumOptimizer()在更新参数时，利用了超参数，参数更新公式是

其中，𝛼为学习率，超参数为𝛽，𝜃为参数，𝑔(𝜃𝑖−1 )为损失函数的梯度。

③tf.train.AdamOptimizer()是利用自适应学习率的优化算法，Adam算法和随机梯度下降算法不同。
随机梯度下降算法保持单一的学习率更新所有的参数，学 习率在训练过程中并不会改变。
而 Adam 算法通过计算梯度的一阶矩估计和二 阶矩估计而为不同的参数设计独立的自适应性学习率。

i.> 学习率：决定每次参数更新的幅度。

优化器中都需要一个叫做学习率的参数，使用时，如果学习率选择过大会出现震荡不收敛的情况，
如果学习率选择过小，会出现收敛速度慢的情况。我们可以选个比较小的值填入，比如 0.01、0.001
"""

# #准确率计算
# prediction = tf.argmax(logits, 1) # 根据axis取值的不同返回每行(1)或者每列(0)最大值的索引
# correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
"""


假设我们手上有60个正样本，40个负样本，我们想找出所有的正样本，模型查找出50个，其中只有40个是真正的正样本
TP: 将正类预测为正类数  40  
FN: 将正类预测为负类数  20 
FP: 将负类预测为正类数  10 
TN: 将负类预测为负类数  30 
准确率(accuracy) = 预测对的/所有 = (TP+TN)/(TP+FN+FP+TN) = 70%
精确率(precision)、查准类= TP/(TP+FP) = 80%  
命中敌人的炮弹数（选出正例的个数）  / 发射的炮弹数（选出样本的个数）     过滤垃圾邮件
召回率(recall)、查全率 = TP/(TP+FN) = 2/3 
命中敌人的炮弹数（选出正例的个数） / 敌人的总数（真实值为正例的个数）   
TPR：在所有实际为阳性的样本中，被正确地判断为阳性之比率。TPR=TP/(TP+FN)  (和召回率一样)
FPR：在所有实际为阴性的样本中，被错误地判断为阳性之比率。FPR=FP/(FP+TN)
ROC曲线
AUC是ROC曲线下的面积


•	模型评估
    •	留出法
        •	按比例分割
    •	交叉验证
        •	分成k份，k-1份训练集，另外的1份是测试集，分别创建k个模型，平均权重
    •	留一法
        •	交叉验证的极端方法，只有1个样本做测试集
        •	计算量大
    •	自助法
        •	将样本有放回抽样n个，执行n，最后产出样本比例接近0.368（e）
•	检测指标
    •	线性回归
        •	MSE
        •	均方误差根
        •	R方
    •	分类
        •	准确率
        •	召回率
        •	ROC
        •	AUC


"""

# #创建会话
# sess = tf.Session()
"""
1.计算图
用户不定义计算图时，系统会自动维护一个默认的计算图,
tensorflow 会自动将定义的所有计算添加到默认的计算图
tf.get_default_graph() # 获取当前的默认图
用户自己创建计算图，用with创建图指定为默认计算图后,
下面的运算都在这个计算图内，变量为该计算图独有，不与其他计算图共享
g1 = tf.Graph()
with g1.as_default() :
    a=tf.get_variable("a",[2],initializer=tf.ones_initializer())
    b=tf.get_variable("b",[2],initializer=tf.zeros_initializer())
获取某个变量的计算图，如变量a
a.graph()
2.张量
张量只是引用了程序中的运算结果而不是一个真正的数组，张量保存的是运算结果的属性，而不是真正的数字
import tensorflow as tf 
a=tf.constant([1.0,2.0],name ="a") 
b=tf.constant([3.0,4.0],name ="b") 
result=a+b 
print(result) 
＃输出 Tensor ("add:0"，shape=(2,), dtype=float32)
add:0:由加法得来的第一个输出
shape=(2,):形状为2的数组，只有一个维度，注意不是2*1，
dtype=float32:元素类型为float32

要想获得result的真值需定义会话sess进行真正的运算
使用sess.run(result)
或result.eval(session=sess)
或sess为默认会话时result.eval()

3.会话(Session)
定义会话
# 接上
with tf.Session() as sess :
    tf.initialize_all_variables() 
    print(sess.run(result))

＃输出［ 4. 6.]
指定sess为默认会话
sess = tf.Session()
with sess.as default():
  <with-block>
在定义计算时tensorflow会自动生成一个默认的计算图，如果没有特殊指定，
定义的运算会自动加入这个计算图中。通过手动指定，
会话也可以成为默认的(tensorflow不会自动生成默认的会话)
Session()中有可指定图的参数 graph=
"""
# sess.run(tf.global_variables_initializer()) #全局变量初始化
"""
变量的初始化必须在模型的其它操作运行之前先明确地完成。
最简单的方法就是添加一个给所有变量初始化的操作，并在使用模型之前首先运行那个操作。

有时候会需要用另一个变量的初始化值给当前变量初始化。
由于tf.global_variables_initializer()是并行地初始化所有变量，所以在有这种需求的情况下需要小心。

用其它变量的值初始化一个新的变量时，使用其它变量的initialized_value()属性。
可以直接把已初始化的值作为新变量的初始值，或者把它当做tensor计算得到一个值赋予新变量。
"""
# #迭代训练
# for step in range(1501):
#     cost_val, _, acc = sess.run([cost, train, accuracy], feed_dict={X: x_data, Y: y_data})
#     if step % 100 == 0:# 显示损失值收敛情况
#         print(step, cost_val, acc)
# #准确率
# h, c, a = sess.run([hypothesis, prediction, accuracy], feed_dict={X: x_data, Y: y_data})
# print("Accuracy: ", a)
# # 测试
# h1, p1 = sess.run([hypothesis, prediction],
#     feed_dict={X: [[0,1,1,0,1,0,0,0,1,1,0,0,2,1,0,0]]})
# print(h1, p1)

