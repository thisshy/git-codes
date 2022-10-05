# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 18:04:54 2022

@author: 11870
"""


# -*- coding: utf8 -*-
import numpy as np

import keras
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score,train_test_split,KFold
from sklearn.preprocessing import LabelEncoder
from keras.layers import Dense,Dropout,Flatten,Conv1D,MaxPooling1D
from keras.models import model_from_json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
 
# 载入数据
# df = np.load('data1.npy')
# X = np.expand_dims(df[:, 0:1000].astype(float), axis=2)
# Y = df[:,-1]


df = np.load('3_5.npy')
X=df[:, 0:1000]*pow(10,23)

X = np.expand_dims(X, axis=2).astype(np.float64)

Y = df[:,-1]


 
# 湿度分类编码为数字
encoder = LabelEncoder()
Y_encoded = encoder.fit_transform(Y)
Y_onehot = np_utils.to_categorical(Y_encoded)
 
# 划分训练集，测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_onehot, test_size=0.3, random_state=0)



np.random.seed(7)
np.random.shuffle(X_train)
np.random.seed(7)
np.random.shuffle(Y_train)
np.random.seed(7)
np.random.shuffle(X_test)
np.random.seed(7)
np.random.shuffle(Y_test)




# 定义神经网络
def baseline_model():
    model = Sequential()
    model.add(Conv1D(16, 3, input_shape=(1000, 1)))
    model.add(Conv1D(16, 3, activation='tanh'))
    # model.add(MaxPooling1D(3))
    model.add(Conv1D(64, 3, activation='tanh'))
    model.add(Conv1D(64, 3, activation='tanh'))
    # model.add(MaxPooling1D(3))
    model.add(Conv1D(128, 3, activation='tanh'))
    # model.add(MaxPooling1D(3))
    model.add(Conv1D(256, 3, activation='tanh'))
    # model.add(MaxPooling1D(3))
    model.add(Conv1D(512, 3, activation='tanh'))
    # model.add(MaxPooling1D(3))
    model.add(Conv1D(1024, 3, activation='tanh'))
    # model.add(MaxPooling1D(3))
    # model.add(Conv1D(2048, 3, activation='tanh'))
    # # model.add(MaxPooling1D(3))
    # model.add(Conv1D(4096, 3, activation='tanh'))
    # # model.add(MaxPooling1D(3))
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))
    print(model.summary())
    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
    return model
 
# 训练分类器
estimator = KerasClassifier(build_fn=baseline_model, epochs=100, batch_size=1, verbose=1)
estimator.fit(X_train, Y_train)
 

 
# 将其模型转换为json
model_json = estimator.model.to_json()
with open(r"model.json", 'w')as json_file:
    json_file.write(model_json)  # 权重不在json中,只保存网络结构
estimator.model.save_weights('model.h5')

# 加载模型用做预测
json_file = open(r"model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
print("loaded model from disk")
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 分类准确率
print("The accuracy of the classification model:")
scores = loaded_model.evaluate(X_test, Y_test, verbose=0)
print('%s: %.2f%%' % (loaded_model.metrics_names[1], scores[1] * 100))

# 输出预测类别
predicted = loaded_model.predict(X_test)  # 返回对应概率值
predicted_label = np.argmax(loaded_model.predict(X_test), axis=1)








