# -*- coding: utf-8 -*-
# @Time    : 2019/9/29 0029 8:34
# @Author  : 
# @FileName: test02.py
# @Software: PyCharm
import tensorflow as tf
# （一）定义变量a，值为18（8分）
a = tf.Variable(18)
b = tf.constant([[2,3,4],[3,4,5]])
sum1 = tf.Variable(0)
# （二）定义变量a乘以2的操作（8分）
mul = tf.multiply(a, 2)
# （三）定义变量a除以3的操作（8分）
div = tf.divide(a, 3)
# （四）创建Session对象（8分）
sess = tf.Session()

# （五）执行全局变量初始化（8分）
sess.run(tf.global_variables_initializer())
# （六）输出变量a的值（8分）
print(sess.run(a))
# （七）执行变量a乘以2的操作，输出结果（8分）
print(sess.run(mul))
# （八）把结果赋值给a（8分）
a = sess.run(mul)
# （九）执行变量a除以3的操作，输出结果（8分）
print(sess.run(div))
# （十）对张量[[2,3,4],[3,4,5]] 执行转置函数，参数用默认值，写出最后的结果（8分）
print(sess.run(tf.transpose(b)))
# （十一）完成100以内的整数的求和运算，输出每个步骤的结果。（每个步骤5分，至少4个步骤，共20分）
for i in range(101):
    sess.run(tf.assign(sum1, sum1+i))
    print(sess.run(sum1))