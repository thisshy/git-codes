# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 16:17:52 2021

@author: 11870
"""
import tensorflow as tf
import numpy as np

# w=tf.Variable(tf.constant(5,dtype=tf.float32))
# lr=0.2
# epoch=40
# for epoch in range(epoch):
#     with tf.GradientTape() as tape:
#         loss=tf.square(w+1)
#     grads=tape.gradient(loss,w)
    
#     w.assign_sub(lr*grads)
#     print('After %s epoch,w is %f,loss is %f' % (epoch,w.numpy(),loss))

# d=tf.random.truncated_normal([3,3],mean=0.3,stddev=1)
# a=tf.random.uniform([3,3],minval=0,maxval=3)
# c=tf.reduce_max(a)
# print(c)
# features=tf.constant([1,2,3,4])
# labels=tf.constant([11,22,33,44])
# dataset=tf.data.Dataset.from_tensor_slices((features,labels))
# print(dataset)
# for element in dataset:
#     print(element)

from sklearn import datasets
from pandas import DataFrame
import pandas as pd
x_data=datasets.load_iris().data
y_data=datasets.load_iris().target
# print('x_data from datasets:\n',x_data)
# print('y_data from datasets:\n',y_data)
# x_data=DataFrame(x_data,columns=['花萼长度','花萼宽度','花瓣长度','花瓣宽度'])
# pd.set_option('display.unicode.east_asian_width', True)
# print('x_data add index: \n',x_data)
# x_data['类别']=y_data
# print('x_data 增加一列标签：\n',x_data)

# np.random.seed(116)
# np.random.shuffle(x_data)
# np.random.seed(116)
# np.random.shuffle(y_data)
# tf.random.set_seed(116)
# x_train=x_data[:-30]
# y_train=y_data[:-30]
# x_test=x_data[-30:]
# y_test=y_data[-30:]
# train_db=tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(32)
# test_db=tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(32)

