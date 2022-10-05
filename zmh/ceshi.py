# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 23:35:11 2022

@author: 11870
"""
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.preprocessing import LabelEncoder



# 载入数据
df = np.load('3_5.npy')
X=df[:, 0:1000]*pow(10,23)

X = np.expand_dims(X, axis=2).astype(np.float32)

Y = df[:,-1]
 
# 湿度分类编码为数字
encoder = LabelEncoder()
Y_encoded = encoder.fit_transform(Y)
Y_onehot = np_utils.to_categorical(Y_encoded)
 
# 划分训练集，测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_onehot, test_size=0.3, random_state=0)
X_train = X_train.reshape(X_train.shape[0], -1)
model = Sequential([
    Dense(100, input_dim=1000),
    Activation('relu'),
    Dense(100, input_dim=1000),
    Activation('relu'),
    Dense(100, input_dim=1000),
    Activation('relu'),
    Dense(2),
    Activation('softmax')
])


rmsprop = keras.optimizers.rmsprop_v2.RMSprop(lr=0.0001)


model.compile(
    optimizer=rmsprop,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print('Training------------')

model.fit(X_train, Y_train,epochs=10, batch_size=32)

print('\nTesting------------')

loss, accuracy = model.evaluate(X_test, Y_test)
a=model.predict(X_test)
print('test loss:', loss)
print('test accuracy:', accuracy)

