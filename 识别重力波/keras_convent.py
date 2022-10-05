# import h5py
import pickle
import keras
from keras.layers import Conv2D,MaxPool2D,Dense,Flatten,Dropout

from keras.models import Sequential 

from keras.utils import to_categorical

from keras.datasets import cifar10

import matplotlib.pyplot as plt

#这是载入打包后的数据，"tranindata.bin"是打包后的数据文件名
dic = pickle.load(open("traindata.bin", "rb"))
X = dic["train"]
Y = dic["train_label"]
X_test = dic["train_test"]
Y_test = dic["train_test_label"]


X /= 255
X_test /= 255
Y = to_categorical(Y,8)
Y_test = to_categorical(Y_test,8)

#data_shape是输入数据的形状
data_shape = (128,128,1)

lenet=Sequential()

#输入层卷积核个数是16个，卷积核是3*3，偏移是1，激活函数是relu
lenet.add(Conv2D(16,kernel_size=3,strides=1,padding='same',activation='relu',input_shape=data_shape))

#最大混合层，前一层输出2*2矩阵内取最大值
lenet.add(MaxPool2D(pool_size=2,strides=2))

#随机丢弃，防止过拟合
lenet.add(Dropout(0.5))

#卷积层，卷积核个数128个，卷积核是5*5，偏移是1
lenet.add(Conv2D(128,kernel_size=5,strides=1,padding='same',activation='relu'))

#最大混合层，前一层输出2*2矩阵内取最大值
lenet.add(MaxPool2D(pool_size=2,strides=2))

#随机丢弃，防止过拟合
lenet.add(Dropout(0.5))

#将结果展开成一维数组
lenet.add(Flatten())

lenet.add(Dropout(0.5))


#全连接层，一共8个神经元
lenet.add(Dense(8,activation='softmax',name="preds"))

#loss函数是categorical_crossentropy
lenet.compile('adam',loss='categorical_crossentropy',metrics=['accuracy'])


#LossHistory这个类是保存每次的loss和准确率
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()

history = LossHistory()

#开始训练，batch_size表示随机采样的数量，epochs是训练的次数
lenet.fit(X,Y,batch_size=64,epochs=30,validation_data=[X_test,Y_test],callbacks=[history])

#显示每次训练的loss和准确率
history.loss_plot('epoch')
#结果保存为mymodel文件
lenet.save('mymodel')
