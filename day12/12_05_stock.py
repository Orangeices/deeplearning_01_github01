# -*- coding: utf-8 -*-
# @Time    : 2019/10/28 0028 9:54
# @Author  : 
# @FileName: 12_05_stock.py
# @Software: PyCharm

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
tf.set_random_seed(111)
def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator+1e-7)

seq_length = 7
data_dim = 5
hidden_dim = 10
output_dim = 1
learning_rate = 0.1
iteration = 100

xy = np.loadtxt('data-02-stock_daily.csv', delimiter=',')
xy = xy[::-1]
x = xy
y = xy[:, [-1]]



