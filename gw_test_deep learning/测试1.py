# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 20:46:02 2022

@author: W10
"""
# import tensorflow

import numpy as np
import matplotlib.pyplot as plt

from keras import  layers, models
from sklearn.model_selection import cross_val_score,train_test_split,KFold
import keras

# 载入数据
df = np.load('3_5.npy')
X=df[:, 0:1000]*pow(10,23)

X = np.expand_dims(X, axis=2).astype(np.float32)

Y = df[:,-1]


# 划分训练集，测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=0)

a=Y_test
np.random.seed(7)
np.random.shuffle(X_train)
np.random.seed(7)
np.random.shuffle(Y_train)
np.random.seed(7)
np.random.shuffle(X_test)
np.random.seed(7)
np.random.shuffle(Y_test)




model = models.Sequential()
model.add(layers.Conv1D(32, 3, activation='relu', input_shape=(1000,1)))
model.add(layers.MaxPooling1D(2))
model.add(layers.Conv1D(64,3, activation='relu'))
model.add(layers.MaxPooling1D(2))
model.add(layers.Conv1D(64,3, activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1))
model.summary()
model.compile(optimizer=keras.optimizers.adam_v2.Adam(0.000001),
              loss=keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=keras.metrics.Accuracy())
history = model.fit(X_train, Y_train, epochs=10, 
                    validation_data=(X_test,Y_test))
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(X_test,Y_test, verbose=2)

