# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 20:54:53 2021

@author: 11870
"""

# import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import datetime
import itertools
from sklearn import preprocessing
import datetime
from sklearn.metrics import mean_squared_error
from math import sqrt
import produce_data
import math
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv1D, MaxPooling1D
from keras.layers.recurrent import LSTM
from keras import losses
from keras import optimizers
from timeit import default_timer as timer

# //////图片的信息标签可以输入中文的代码如下
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams['axes.unicode_minus'] = False


df= pd.read_csv("test.csv")
df.head()
data = df.values
zz=np.hsplit(data,(1,))[0]
# data_x=np.hsplit(data,(3,))[0]
# data_y=np.hsplit(data,(3,))[1]
# len1=int(0.8*len(data))
# x_train=np.vsplit(data,(len1,))[0].reshape(2,4,6551)
# x_test=np.vsplit(data,(len1,))[1].reshape(1638,2,4)
# y_train=np.vsplit(data_y,(len1,))[0]
# y_test=np.vsplit(data_y,(len1,))[1]
result=[]
time_steps = 3
 
for i in range(len(data)-time_steps):
    result.append(data[i:i+time_steps])
 
result=np.array(result)
 
#训练集和测试集的数据量划分
train_size = int(0.8*len(result))
train = result[:train_size]
x_train = result[:,:]
y_train = result[:,-1][:,-1]
x_test = result[train_size:,:]
y_test = result[train_size:,-1][:,-1]

###############################################构建模型
def build_model(input):
    model = Sequential()
    model.add(Dense(50, input_shape=(input[0], input[1])))
    model.add(Conv1D(filters=128, kernel_size=1, padding='valid', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling1D(pool_size=2, padding='valid'))
    model.add(Conv1D(filters=64, kernel_size=1, padding='valid', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling1D(pool_size=1, padding='valid'))
    # model.add(Conv1D(filters=32, kernel_size=1, padding='valid', activation='relu', kernel_initializer='uniform'))
    # model.add(MaxPooling1D(pool_size=1, padding='valid'))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='uniform'))
    model.add(Dense(1, activation='relu', kernel_initializer='uniform'))
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    return model
 
 
model = build_model([3, 4, 1])
print(model.summary())
 
################################################ 训练数据预测

start = timer()
history = model.fit(x_train,
                    y_train,
                    batch_size=128,
                    epochs=20,
                    validation_split=0.1,
                    verbose=2)
end = timer()
print(end - start)

history_dict = history.history
history_dict.keys()
aa=produce_data.re_maxv()


############################################### 保存训练参数
# # print(model.trainable_variables)
# file = open('./weights.txt', 'w')
# for v in model.trainable_variables:
#     file.write(str(v.name) + '\n')
#     file.write(str(v.shape) + '\n')
#     file.write(str(v.numpy()) + '\n')
# file.close()

###############################################    show   ###############################################
#                                       均方误差
 
plt.subplot(1,3,1)
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
loss_values50 = loss_values[0:150]
val_loss_values50 = val_loss_values[0:150]
epochs = range(1, len(loss_values50) + 1)
plt.plot(epochs, loss_values50, 'b', color='blue', label='Training loss')
plt.plot(epochs, val_loss_values50, 'b', color='red', label='Validation loss')
plt.rc('font', size=18)
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('均方误差')
plt.legend()
# plt.xticks(epochs)
# fig = plt.gcf()
# fig.set_size_inches(15, 7)
# fig.savefig('img/tcstest&validationlosscnn.png', dpi=300)

 
###############################################平均绝对误差
# plt.subplot(1,3,2)
# print(history.history.keys())
# acc=history.history['mae']
# val_acc=history.history['val_mae']
# plt.plot(acc,label='acc')
# plt.plot(val_acc,label='val_acc')
# plt.title('acc and valacc')
# plt.xlabel('Epochs')
# plt.ylabel('平均绝对误差')
# plt.legend()


plt.subplot(1,3,3)
# ################################################ 真实值和预测值
p = aa*model.predict(x_test)
plt.scatter(np.arange(len(p)),y_test*aa,label='test',s=15)
plt.scatter(np.arange(len(p)),p,label='预测值',s=15)
# cc=abs(p-y_test)
# plt.plot(np.arange(len(p)),cc)
plt.legend(loc='lower right',fontsize=20)
plt.show()

# z=zz[len(zz)-len(p):]

# plt.scatter(z,y_test*aa,label='test',s=15)
# plt.scatter(z,p,label='预测值',s=15)


