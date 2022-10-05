# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 14:36:00 2022

@author: W10
"""

import matplotlib.pyplot as plt
from math import sqrt
from matplotlib import pyplot
import pandas as pd
import numpy as np
from numpy import concatenate
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras import layers
import keras
import tensorflow as tf

'''Keras实现神经网络回归模型'''
# 读取数据



values =np.load('data7_2800.npy',allow_pickle=True).astype(np.float32)
# # 原始数据标准化，为了加速收敛
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

n1=600
n2=2800
n3=4481
Y = scaled[:, -1]
X = scaled[:,0:-1]
c=values[:, -1]
train_x = X[0:n1]
train_y = Y[0:n1]
test_x = X[n1:n2]
test_y = Y[n1:n2]


"""
输入要求：[送入样本数， 循环核时间展开步数， 每个时间步输入特征个数]
"""
train_x = np.reshape(train_x, (train_x.shape[0], 1, n3))
test_x = np.reshape(test_x,  (test_x.shape[0], 1, n3))

# 创建模型
model = Sequential()
# 循环神经网络
# 隐藏层100
model.add(layers.LSTM(units=2000, return_sequences=True))
model.add(Activation('relu'))
model.add(Dropout(0.1))

# # 隐藏层100
# model.add(layers.GRU(units=1000,return_sequences=True))
# model.add(Activation('relu'))
# model.add(Dropout(0.25))

# # 隐藏层100
# model.add(layers.LSTM(units=1000,return_sequences=True))
# model.add(Activation('relu'))
# model.add(Dropout(0.1))

# 隐藏层100
model.add(layers.LSTM(units=2000))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(1))



# 编译模型
# 使用高效的ADAM优化算法以及优化的最小均方误差损失函数
model.compile(loss='mean_squared_error', optimizer= keras.optimizers.adam_v2.Adam(lr=0.00009))

# 训练模型
history = model.fit(train_x, train_y, epochs=10, batch_size=32,
                    validation_data=(test_x, test_y), verbose=1)
model.summary()  # 查看你的神经网络的架构和参数量等信息


# 预测
yhat = model.predict(test_x)
test_x = np.reshape(test_x,  (test_x.shape[0],n3))
# 预测y 逆标准化
inv_yhat0 = concatenate((test_x, yhat), axis=1)
inv_yhat1 = scaler.inverse_transform(inv_yhat0)
inv_yhat = inv_yhat1[:, -1].reshape(len(test_y),1)


# 预测 逆
aa=model.predict(train_x)
train_x = np.reshape(train_x,  (train_x.shape[0],n3))
inv_y0 = concatenate((train_x,aa), axis=1)
inv_y1 = scaler.inverse_transform(inv_y0)
inv_y = inv_y1[:, -1].reshape(len(train_y),1)

# 计算 R2
r_2 = r2_score(test_y, yhat)
print('Test r_2: %.3f' % r_2)
# 计算MAE
mae = mean_absolute_error(test_y, yhat)
print('Test MAE: %.3f' % mae)
# 计算RMSE
rmse = sqrt(mean_squared_error(test_y, yhat))
print('Test RMSE: %.3f' % rmse)



b=np.vstack((inv_y,inv_yhat))


plt.plot(b)
plt.plot(c)

