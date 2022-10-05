"打包数据来训练"
import numpy as np
import pickle
from PIL import Image
import glob
import random
from skimage import feature as ft

#只有星星a
#有点泛白有星星b
#一条光线有星星c
#全黑没星星d
#光球有星星e
#纯白图片f
#超级亮没星星g
#超级亮可能有星星h



a_path = "./分类完整图片/训练集/只有星星/*.png"
b_path = "./分类完整图片/训练集/有点泛白有星星/*.png"
c_path = "./分类完整图片/训练集/一条光线有星星/*.png"
d_path = "./分类完整图片/训练集/全黑没星星/*.png"
e_path = "./分类完整图片/训练集/光球有星星/*.png"
f_path = "./分类完整图片/训练集/纯白图片/*.png"
g_path = "./分类完整图片/训练集/超级亮没星星/*.png"
h_path = "./分类完整图片/训练集/超级亮可能有星星/*.png"


a_test_path = "./分类完整图片/测试集/只有星星test/*.png"
b_test_path = "./分类完整图片/测试集/有点泛白有星星test/*.png"
c_test_path = "./分类完整图片/测试集/一条光线有星星test/*.png"
d_test_path = "./分类完整图片/测试集/全黑没星星test/*.png"
e_test_path = "./分类完整图片/测试集/光球有星星test/*.png"
f_test_path = "./分类完整图片/测试集/纯白图片test/*.png"
g_test_path = "./分类完整图片/测试集/超级亮没星星test/*.png"
h_test_path = "./分类完整图片/测试集/超级亮可能有星星test/*.png"


#训练数据保存到traindata.bin
train_path = "./traindata.bin"


size = (128,128)
resize = [128,128,1]

#加下面这个函数，因为高清大图是16位图，转成8位图比较好
def imgDepth16to8(imgArray):
    
    imgArray /= 256

    imgArray=imgArray.astype(np.int32)
    imgArray=imgArray.astype(np.float32)
    return imgArray

#将图片分别旋转90,180,270度
def createThreeimg(img):
    img1 = img.rotate(90)
    img2 = img.rotate(180)
    img3 = img.rotate(270)
    array1 = np.array(img1,np.float32).reshape(resize)  #把图片变成数组，size不变。
    array1 = imgDepth16to8(array1)     #线性缩小到8位格式。

    array2 = np.array(img2,np.float32).reshape(resize)
    array2 = imgDepth16to8(array2)

    array3 = np.array(img3,np.float32).reshape(resize)
    array3 = imgDepth16to8(array3)


    return [array1,array2,array3]

def createImgList(pa):
    li = [[]]
    for s in glob.glob(pa):
        img = Image.open(s)
        img.thumbnail(size, Image.ANTIALIAS)
        array = np.array(img,np.float32).reshape(resize)
        array = imgDepth16to8(array)

        arratL = createThreeimg(img)
        if(li==[[]]):
            li = [array]
            for a in arratL:
                li = np.concatenate((li,[a]),axis=0)
            continue
        else:
            li= np.concatenate((li,[array]),axis=0)
            for a in arratL:
                li = np.concatenate((li,[a]),axis=0)
    return li

def createLabelList(li):
    L = np.array([])

    for index in range(len(li)):
        list_num = len(li[index])
        temp = np.ones(list_num,np.float32)*index
        L = np.concatenate((L,temp),axis=0)

    return L

a_list = createImgList(a_path)
b_list = createImgList(b_path)
c_list = createImgList(c_path)
d_list = createImgList(d_path)
e_list = createImgList(e_path)
f_list = createImgList(f_path)
g_list = createImgList(g_path)
h_list = createImgList(h_path)

a_test_list = createImgList(a_test_path)
b_test_list = createImgList(b_test_path)
c_test_list = createImgList(c_test_path)
d_test_list = createImgList(d_test_path)
e_test_list = createImgList(e_test_path)
f_test_list = createImgList(f_test_path)
g_test_list = createImgList(g_test_path)
h_test_list = createImgList(h_test_path)

li1 = [a_list,b_list,c_list,d_list,e_list,f_list,g_list,h_list]
li2 = [a_test_list,b_test_list,c_test_list,d_test_list,e_test_list,f_test_list,g_test_list,h_test_list]

train_label = createLabelList(li1)
train_test_label = createLabelList(li2)

train_list = np.concatenate((a_list,b_list),axis=0)
train_list = np.concatenate((train_list,c_list),axis=0)
train_list = np.concatenate((train_list,d_list),axis=0)
train_list = np.concatenate((train_list,e_list),axis=0)
train_list = np.concatenate((train_list,f_list),axis=0)
train_list = np.concatenate((train_list,g_list),axis=0)
train_list = np.concatenate((train_list,h_list),axis=0)

train_test_list = np.concatenate((a_test_list,b_test_list),axis=0)
train_test_list = np.concatenate((train_test_list,c_test_list),axis=0)
train_test_list = np.concatenate((train_test_list,d_test_list),axis=0)
train_test_list = np.concatenate((train_test_list,e_test_list),axis=0)
train_test_list = np.concatenate((train_test_list,f_test_list),axis=0)
train_test_list = np.concatenate((train_test_list,g_test_list),axis=0)
train_test_list = np.concatenate((train_test_list,h_test_list),axis=0)




print(train_list.shape)
print(train_label.shape)

print(train_test_list.shape)
print(train_test_label.shape)

#打包数据，train保存训练数据，train_label是训练数据的标签，train_test是测试集的数据，train_test_label是测试集数据的标签
dic = {
    "train":train_list,
    "train_label":train_label,
    "train_test":train_test_list,
    "train_test_label":train_test_label
    }

f = open(train_path,"wb")
pickle.dump(dic,f)
