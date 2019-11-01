# -*- coding: utf-8 -*-
# @Time    : 2019/10/16 0016 8:58
# @Author  : 
# @FileName: ckpt_demo.py
# @Software: PyCharm

import os
from tensorflow.python import pywrap_tensorflow

# code for finall ckpt
# checkpoint_path = os.path.join('~/tensorflowTraining/ResNet/model', "model.ckpt")

#code for designated ckpt, change 3890 to your num
# checkpoint_path = os.path.join('./ckpt/cnn')
# # Read data from checkpoint file
# reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
# var_to_shape_map = reader.get_variable_to_shape_map()
# # Print tensor name and values
# for key in var_to_shape_map:
#     print("tensor_name: ", key)
    # print(reader.get_tensor(key))

def read_tensor_name(filenmae):
    import os
    from tensorflow.python import pywrap_tensorflow
    keys = []
    ckpt_path = os.path.join(filenmae)
    print(ckpt_path)
    reader = pywrap_tensorflow.NewCheckpointReader(ckpt_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in var_to_shape_map:
        keys.append(key)
        print("tensor_name: ", key)

read_tensor_name('../day09/ckpt/cnn_file')

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

detection_graph = tf.Graph()


learning_rate = 0.001
train_epochs = 15
batch_size = 100
with tf.Session(graph=detection_graph) as sess:
    # 读取模型
    loader = tf.train.import_meta_graph('./ckpt/cnn.meta')
    loader.restore(sess, tf.train.latest_checkpoint('./ckpt'))

    # 一些中间过程，比如生成、处理input_data等
    ...

    # 使用模型
    input_tensor0 = detection_graph.get_tensor_by_name('Variable:0')
    input_tensor1 = detection_graph.get_tensor_by_name('Variable_1:0')
    input_tensor2 = detection_graph.get_tensor_by_name('w3:0')

# ————————————————
# 版权声明：本文为CSDN博主「umbrellalalalala」的原创文章，遵循
# CC
# 4.0
# BY - SA
# 版权协议，转载请附上原文出处链接及本声明。
# 原文链接：https: // blog.csdn.net / umbrellalalalala / article / details / 88826085