# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 14:10:06 2021

@author: 11870
"""

from __future__ import print_function
import  pandas as  pd
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import itertools
from sklearn import preprocessing
import datetime
from sklearn.metrics import mean_squared_error

  

data=np.load('data3_1.npy')


# 数据切分
result=[]
result1=[]
time_steps = 3

for i in range(len(data)-time_steps):
    result.append(data[i:i+time_steps])

result=np.array(result)
 
#训练集和测试集的数据量划分
train_size = int(0.8*len(result))

#训练集切分
train = result[:train_size,:]
 
x_train = train[:,:,:-1]
y_train = train[:,-1][:,-1]
 
x_test = result[train_size:,:,:-1]
y_test = result[train_size:,-1][:,-1]
np.random.seed(7)
np.random.shuffle(x_train)
np.random.seed(7)
np.random.shuffle(y_train)
tf.random.set_seed(7)


#数据重塑

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2])
 

 
#模型构建
import math
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv1D, MaxPooling1D
from keras.layers.recurrent import LSTM
from keras import losses
from keras import optimizers
 
def build_model(input):
    model = Sequential()
    model.add(Dense(20, input_shape=(input[0], input[1])))
    model.add(Conv1D(filters=64, kernel_size=1, padding='valid', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling1D(pool_size=2, padding='valid'))
    model.add(Conv1D(filters=64, kernel_size=1, padding='valid', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling1D(pool_size=1, padding='valid'))
    model.add(Dropout(0.20))
    model.add(Conv1D(filters=32, kernel_size=1, padding='valid', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling1D(pool_size=1, padding='valid'))
    model.add(Dropout(0.20))
    model.add(Conv1D(filters=32, kernel_size=1, padding='valid', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling1D(pool_size=1, padding='valid'))
    model.add(Dropout(0.20))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='uniform'))
    model.add(Dense(1, activation='relu', kernel_initializer='uniform'))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model
 

model = build_model([3, 1000, 1])
 
# Summary of the Model
print(model.summary())
 
# 训练数据预测
from timeit import default_timer as timer
start = timer()
history = model.fit(x_train,
                    y_train,
                    batch_size=128,
                    epochs=100,
                    validation_split=0.2,
                    verbose=2)
end = timer()
print(end - start)
 
# 画出训练集和验证集的损失曲线
# epochs=1

plt.subplot(1,3,1)
loss_values = history.history['loss']
val_loss_values = history.history['val_loss']

plt.plot(loss_values, 'b', color='blue', label='Training loss')
plt.plot(val_loss_values, 'b', color='red', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('均方误差')
plt.legend()


 
plt.subplot(1,3,2)
print(history.history.keys())
acc=history.history['mae']
val_acc=history.history['val_mae']
plt.plot(acc,label='acc')
plt.plot(val_acc,label='val_acc')
plt.title('acc and valacc')
plt.xlabel('Epochs')
plt.ylabel('平均绝对误差')
plt.legend()


plt.subplot(1,3,3)
# # 画出真实值和测试集的预测值之间的对比图像
p = model.predict(x_test)
plt.scatter(np.arange(len(p)),y_test,label='test',s=15)
plt.scatter(np.arange(len(p)),p,label='预测值',s=15)
plt.legend(loc='lower right',fontsize=20)
# plt.xlabel('No. of Trading Days')
# plt.ylabel('Close Value (scaled)')
# fig = plt.gcf()
# fig.set_size_inches(15, 5)
# #fig.savefig('img/tcstestcnn.png', dpi=300)
# plt.show()
plt.show()
