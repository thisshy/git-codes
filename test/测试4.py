# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 20:34:47 2022

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
import os
'''Keras实现神经网络回归模型'''
# 读取数据



values =np.load('data9_1000_3009.npy',allow_pickle=True).astype(np.float32)
# # 原始数据标准化，为了加速收敛
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled = scaler.fit_transform(values)
Y = values[:, -1]
X = values[:,0:-1]

train_x = X[0:800]
train_y = Y[0:800]
test_x = X[800:1000]
test_y = Y[800:1000]

input = X.shape[1]
time_steps = input

# train_x, train_y = np.array(train_x), np.array(train_y)
# test_x, test_y = np.array(test_x), np.array(test_y)

# print(train_x.shape)
"""
输入要求：[送入样本数， 循环核时间展开步数， 每个时间步输入特征个数]
"""
train_x = np.reshape(train_x, (train_x.shape[0], 1, 4482))
test_x = np.reshape(test_x,  (test_x.shape[0], 1, 4482))

# 创建模型
model = Sequential()
# 循环神经网络
# 隐藏层100
model.add(layers.LSTM(units=1300, return_sequences=True))
model.add(Activation('relu'))
model.add(Dropout(0.1))

# # 隐藏层100
# model.add(layers.GRU(units=100,return_sequences=True))
# model.add(Activation('relu'))
# model.add(Dropout(0.2))

# # 隐藏层100
# model.add(layers.LSTM(units=800,return_sequences=True))
# model.add(Activation('relu'))
# model.add(Dropout(0.1))

# # 隐藏层100
# model.add(layers.LSTM(units=800,return_sequences=True))
# model.add(Activation('relu'))
# model.add(Dropout(0.1))


# 隐藏层100
model.add(layers.LSTM(units=1300))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(1))



# 编译模型
# 使用高效的ADAM优化算法以及优化的最小均方误差损失函数
model.compile(loss='mean_squared_error', optimizer= keras.optimizers.adam_v2.Adam(lr=0.0001))

# 训练模型
# history = model.fit(train_x, train_y, epochs=1, batch_size=32,
#                     validation_data=(test_x, test_y), verbose=1,callbacks=[cp_callback])
# model.summary()  # 查看你的神经网络的架构和参数量等信息


checkpoint_save_path = "./checkpoint/stock.ckpt"

if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 monitor='val_loss')

history = model.fit(train_x, train_y, epochs=30, batch_size=32,
                    validation_data=(test_x, test_y), verbose=1,callbacks=[cp_callback])
model.summary()  # 查看你的神经网络的架构和参数量等信息

file = open('./weights.txt', 'w')  # 参数提取
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()





# 预测
yhat =model.predict(test_x).reshape(-1)

test_x = np.reshape(test_x,  (test_x.shape[0], 4482))





# 计算 R2

r_2 = r2_score(test_y, yhat)
print('Test r_2: %.3f' % r_2)
# 计算MAE
mae = mean_absolute_error(test_y, yhat)
print('Test MAE: %.3f' % mae)
# 计算RMSE
rmse = sqrt(mean_squared_error(test_y, yhat))
print('Test RMSE: %.3f' % rmse)
plt.plot(test_y,color='red')
plt.plot(yhat,color='green')
plt.title('Intensity Prediction')
plt.show()



train_x = np.reshape(train_x, (train_x.shape[0], 1, 4482))
aa=model.predict(train_x).reshape(800,1)
yhat=yhat.reshape(200,1)
# yhat.reshape(-1)
b=np.vstack((aa,yhat))



