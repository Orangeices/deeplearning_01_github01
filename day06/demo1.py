# -*- coding: utf-8 -*-
# @Time    : 2019/10/14 0014 20:23
# @Author  : 
# @FileName: demo1.py
# @Software: PyCharm
import matplotlib.pylab as plt
import numpy as np

path = r'E:\bawei\DeepLearning_1\深度一\tensorflow补充'
img = plt.imread(path+r'\1.bmp')
gravity = np.array([1., 0., 0.])
greyimg = np.dot(255-img, gravity)/255
print(img.shape)