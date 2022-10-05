import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, GRU
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

# //////图片的信息标签可以输入中文的代码如下
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams['axes.unicode_minus'] = False


df= pd.read_csv("1.csv")
df.head()
data = df.values

df1= pd.read_csv("2.csv")
df1.head()
data1 = df1.values

# data_x=np.hsplit(data,(3,))[0]
# data_y=np.hsplit(data,(3,))[1]
# len1=int(0.8*len(data))
# x_train=np.vsplit(data,(len1,))[0].reshape(2,4,6551)
# x_test=np.vsplit(data,(len1,))[1].reshape(1638,2,4)
# y_train=np.vsplit(data_y,(len1,))[0]
# y_test=np.vsplit(data_y,(len1,))[1]
result=[]
result1=[]
time_steps = 3
 
for i in range(len(data)-time_steps):
    result.append(data[i:i+time_steps])
 
result=np.array(result)

for i in range(len(data1)-time_steps):
    result1.append(data1[i:i+time_steps])
 
result1=np.array(result1)

#训练集和测试集的数据量划分
train_size = int(0.8*len(result))
train = result[:train_size]
x_train = train[:,:,:-1]
y_train = train[:,-1][:,-1]
x_test = result[train_size:,:,:-1]
y_test = result[train_size:,-1][:,-1]
np.random.seed(7)
np.random.shuffle(x_train)
np.random.seed(7)
np.random.shuffle(y_train)
tf.random.set_seed(7)

x1_test=result1[:,:,:-1]
y1_test=result1[:,-1][:,-1]
np.random.seed(7)

np.random.shuffle(x1_test)
np.random.seed(70)
np.random.shuffle(x_test)
np.random.seed(70)
np.random.shuffle(y_test)
np.random.seed(70)
np.random.shuffle(y1_test)
tf.random.set_seed(70)



# # 归一化
# sc = MinMaxScaler(feature_range=(0, 1))  # 定义归一化：归一化到(0，1)之间
# training_set_scaled = sc.fit_transform(training_set)  # 求得训练集的最大值，最小值这些训练集固有的属性，并在训练集上进行归一化
# test_set = sc.transform(test_set)  # 利用训练集的属性对测试集进行归一化


model = tf.keras.Sequential([
    GRU(300, return_sequences=True),
    Dropout(0.2),
    # GRU(600, return_sequences=True),
    # Dropout(0.2),
    GRU(200),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='mse')  # 损失函数用均方误差
# 该应用只观测loss数值，不观测准确率，所以删去metrics选项，一会在每个epoch迭代显示时只显示loss值

checkpoint_save_path = "./checkpoint/stock.ckpt"

if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True,
                                                 monitor='val_loss')

history = model.fit(x_train, y_train, batch_size=64, epochs=10, validation_split=0.1, validation_freq=1,
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
plt.subplot(1,2,1)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

################## predict ######################
# 测试集输入模型进行预测
predict_X = model.predict(x1_test)
# # 对预测数据还原---从（0，1）反归一化到原始范围
# predicted_stock_price = sc.inverse_transform(predicted_stock_price)
# # 对真实数据还原---从（0，1）反归一化到原始范围
# real_stock_price = sc.inverse_transform(test_set[60:])
# 画出真实数据和预测数据的对比曲线
plt.subplot(1,2,2)
plt.scatter(np.arange(len(y1_test)),y1_test, color='red', label='test')
plt.scatter(np.arange(len(y1_test)),predict_X, color='blue', label='predicted')
# plt.plot(np.arange(len(y1_test)),y1_test, color='red', label='test')
# plt.plot(np.arange(len(y1_test)),predict_X, color='blue', label='predicted')
plt.title('test and predict')
plt.xlabel('点数')
plt.ylabel('归一化的振幅')
plt.legend()
plt.show()

# ##########evaluate##############
# calculate MSE 均方误差 ---> E[(预测值-真实值)^2] (预测值减真实值求平方后求均值)
mse = mean_squared_error(predict_X, y1_test)
# calculate RMSE 均方根误差--->sqrt[MSE]    (对均方误差开方)
rmse = math.sqrt(mean_squared_error(predict_X, y1_test))
# calculate MAE 平均绝对误差----->E[|预测值-真实值|](预测值减真实值求绝对值后求均值）
mae = mean_absolute_error(predict_X, y1_test)
print('均方误差: %.6f' % mse)
print('均方根误差: %.6f' % rmse)
print('平均绝对误差: %.6f' % mae)
