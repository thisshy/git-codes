# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 15:47:30 2021

@author: 11870
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, GRU,LSTM,SimpleRNN
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

# //////图片的信息标签可以输入中文的代码如下
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams['axes.unicode_minus'] = False



data=np.load('data9_1000_3009.npy',allow_pickle=True)

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
# np.random.seed(7)
# np.random.shuffle(x_train)
# np.random.seed(7)
# np.random.shuffle(y_train)
# np.random.seed(7)
# np.random.shuffle(x_test)
# np.random.seed(7)
# np.random.shuffle(y_test)
# tf.random.set_seed(7)


#数据重塑

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2]).astype(np.float32)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2]).astype(np.float32)



model = tf.keras.Sequential([
    LSTM(1000, return_sequences=True),
    Dropout(0.1),
    # LSTM(1500, return_sequences=True),
    # Dropout(0.2),
    # GRU(100, return_sequences=True),
    # Dropout(0.1),
    # GRU(100, return_sequences=True),
    # Dropout(0.2),
    # GRU(120, return_sequences=True),
    # Dropout(0.2),
    # GRU(120, return_sequences=True),
    # Dropout(0.2),
    # GRU(120, return_sequences=True),
    # Dropout(0.2),
    LSTM(1000),
    Dropout(0.1),
    Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
              loss='mean_squared_error')  # 损失函数用均方误差
# 该应用只观测loss数值，不观测准确率，所以删去metrics选项，一会在每个epoch迭代显示时只显示loss值

checkpoint_save_path = "./checkpoint/stock.ckpt"

if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 monitor='val_loss')

history = model.fit(x_train.astype(np.float32), y_train.astype(np.float32), batch_size=32, epochs=10, validation_data=(x_test.astype(np.float32),y_test.astype(np.float32)), validation_freq=1,
                    callbacks=[cp_callback])

model.summary()

file = open('./weights.txt', 'w')  # 参数提取
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

loss = history.history['loss']
val_loss = history.history['val_loss']


plt.subplot(2,1,1)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('loss')
plt.ylabel('epoch')
plt.legend()


################## predict ######################
# # 测试集输入模型进行预测
# predicted_stock_price = model.predict(x_test)
# # 对预测数据还原---从（0，1）反归一化到原始范围
# predicted_stock_price = sc.inverse_transform(predicted_stock_price)
# # 对真实数据还原---从（0，1）反归一化到原始范围
# real_stock_price = sc.inverse_transform(test_set[60:])
# 画出真实数据和预测数据的对比曲线
plt.subplot(2,1,2)
p= model.predict(x_test)
plt.plot(y_test, color='red', label='real')
plt.plot(p, color='blue', label='predict')
plt.title('GRU')
plt.xlabel('点数')
plt.ylabel('振幅')
plt.legend()
plt.show()

##########evaluate##############
# calculate MSE 均方误差 ---> E[(预测值-真实值)^2] (预测值减真实值求平方后求均值)
mse = mean_squared_error(p,y_test)
# calculate RMSE 均方根误差--->sqrt[MSE]    (对均方误差开方)
rmse = math.sqrt(mean_squared_error(p,y_test))
# calculate MAE 平均绝对误差----->E[|预测值-真实值|](预测值减真实值求绝对值后求均值）
mae = mean_absolute_error(p,y_test)
print('均方误差: %.6f' % mse)
print('均方根误差: %.6f' % rmse)
print('平均绝对误差: %.6f' % mae)


p1=model.predict(x_train)

yucezhi=np.vstack((p1,p))