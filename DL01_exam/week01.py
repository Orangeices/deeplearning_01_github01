# -*- coding: utf-8 -*-
# @Time    : 2019/10/9 0009 13:35
# @Author  : 
# @FileName: week01.py
# @Software: PyCharm

# ①　导入必要的依赖库。（8分）
import tensorflow as tf
# ②　定义常量a=5。（8分）
a = tf.constant(5)
# ③　定义常量b=8。（8分）
b = tf.constant(8)
# ④　创建会话。（8分）
sess = tf.Session()
# ⑤　初始化变量（8分）
sess.run(tf.global_variables_initializer())
# ⑥　计算a加b并打印输出加法计算结果（8分）
print('a加b:', sess.run(tf.add(a, b)))
# ⑦　计算a减b并打印输出减法计算结果。（8分）
print('a减b:', sess.run(tf.subtract(a, b)))
# ⑧　计算a乘b并打印乘法计算结果。（8分）
print('a乘b:', sess.run(tf.multiply(a, b)))
# ⑨　计算a除b并打印除法计算结果。（8分）
print('a除b:', sess.run(tf.divide(a, b)))
# ⑩　定义一个常量c,其值为[[2,5,4],[1,3,6]] （8分）
c = tf.constant([[2,5,4],[1,3,6]])
# 11　对常量c按行求和并打印输出结果（8分）
print('对常量c按行求和:', sess.run(tf.reduce_sum(c, axis=1)))
# 12　对常量c按列求和并打印输出结果（8分）
print('对常量c按列求和:', sess.run(tf.reduce_sum(c, axis=0)))
# 13　将常量c求总和（4分）
print('常量c求总和:', sess.run(tf.reduce_sum(c)))
sess.close()