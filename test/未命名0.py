# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 12:52:18 2021

@author: 11870
"""
from keras.datasets import boston_housing
import pandas as pd
import numpy as np
# (train_data, train_targets), (test_data, test_targets) =  boston_housing.load_data()
#数据标准化
#由于数据集有13个参考的维度，而这些维度的数据指标的单位是不同的，所以要把这些数据单位指标的影响去除，使数据能够在同一个量纲上进行讨论。
#而就算去除了数据单位，数据之间的关系仍在。
#这里使用的是0均值标准化，即对于输入数据的每个特征(输入数据矩阵中的列),减去特征平均值，再除以标准差，这样得到的特征平均值为0，标准差为1
#如数据为1,2,3，将1,2,3分别减去平均值得-1,0,1。-1,0,1的标准差为√2/√3,再将-1,0,1除去√2/√3,得到-√6/2,0,√6/2;-√6/2,0,√6/2的标准差
#为√((6/4 + 0 + 6/4)/3) = √1 = 1。所以经过标准化，最终得到特征平均值为0，标准差为1的标准正态分布。
df= pd.read_csv("1.csv")
df.head()
train_data = df.values
train_data=np.array(train_data)

df1= pd.read_csv("3.csv")
df1.head()
train_targets= df1.values
train_targets=np.array(train_targets)

df2= pd.read_csv("2.csv")
df2.head()
test_data = df2.values
test_data=np.array(test_data)

df3= pd.read_csv("4.csv")
df3.head()
test_targets = df3.values
test_targets=np.array(test_targets)


mean = train_data.mean(axis=0) #axis=0表示每一列的数据
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
#用于测试数据集标准化的标准差和均值都是从训练数据上得到的，而不能使用在测试数据上计算得到的任何结果
test_data -= mean
test_data /= std
#对于train_data.mean()

from keras import models
from keras import layers
 
def build_model():
    # Because we will need to instantiate
    # the same model multiple times,
    # we use a function to construct it.
    model = models.Sequential()
    #input_shape为传入一个shape给第一层，为13行的矩阵,所以要求输入的数据为13列的矩阵
    model.add(layers.Dense(30, activation='relu',
                            input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))


    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


import numpy as np
k = 4
num_val_samples = len(train_data) // k #整数除法
num_epochs = 200
all_mae_histories = []

for i in range(k):
    print('processing fold #', i)
    #依次把k分数据中的每一份作为校验数据集
    val_data = train_data[i * num_val_samples : (i+1) * num_val_samples]
    val_targets = train_targets[i* num_val_samples : (i+1) * num_val_samples]

    #把剩下的k-1分数据作为训练数据,如果第i分数据作为校验数据，那么把前i-1份和第i份之后的数据连起来
    partial_train_data = np.concatenate([train_data[: i * num_val_samples], 
                                          train_data[(i+1) * num_val_samples:]], axis = 0)
    partial_train_targets = np.concatenate([train_targets[: i * num_val_samples], 
                                            train_targets[(i+1) * num_val_samples: ]],
                                          axis = 0)
    print("build model")
    model = build_model()
    #把分割好的训练数据和校验数据输入网络
    history = model.fit(partial_train_data, partial_train_targets, 
              validation_data=(val_data, val_targets),
              epochs = num_epochs, 
              batch_size = 1, verbose = 0)
    print(history.history.keys())
    mae_history = history.history['val_mae']
    all_mae_histories.append(mae_history) 

import matplotlib.pyplot as plt
p=model.predict(test_data)
plt.plot(p)
plt.plot(test_data)


# a=np.load('h(f)的一组数据.npy')
# b=np.array(a)
# np.savetxt('001.txt',b)
