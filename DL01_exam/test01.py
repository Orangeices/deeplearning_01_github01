# -*- coding: utf-8 -*-
# @Time    : 2019/9/28 0028 8:34
# @Author  : 
# @FileName: test01.py
# @Software: PyCharm
import tensorflow as tf
# 1.	定义一个常量c1，值为小数3.3（8分）
c1 = tf.constant(3.3)
# 2.	定义一个变量a，值为整数10（8分）
a = tf.Variable(10.0)
# 3.	定义另一个变量b，值为5.6（8分）
b = tf.Variable(5.6)
# 4.	定义变量a与常量c1的和的操作（8分）
sum1 = tf.add(a, c1)
# 5.	定义两个变量a和b的和的操作（8分）
sum2 = tf.add(a, b)
# 6.	定义两个变量a和b的差的操作（8分）
sub = tf.subtract(a, b)
# 7.	创建Session对象（8分）
sess = tf.Session()
# 8.	执行全局变量初始化（8分）
sess.run(tf.global_variables_initializer())
# 9.	输出变量a的值（6分）
print(sess.run(a))
# 10.	输出变量b的值（6分）
print(sess.run(b))
# 11.	输出变量a与常量c1的和的值（8分）
print(sess.run(sum1))
# 12.	输出两个变量a和b的和的值（8分）
print(sess.run(sum2))
# 13.	输出两个变量a和b的差的值（8分）
print(sess.run(sub))
sess.close()