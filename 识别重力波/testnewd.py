import numpy as np

from keras.models import load_model
# from keras.preprocessing import image

import pickle
import glob

from PIL import Image

import shutil
import os

# from skimage import feature as ft

#载入训练好的模型，mymodel是模型的名字
model = load_model("./mymodel")


def imgDepth16to8(imgArray):
    
    imgArray /= 256

    imgArray=imgArray.astype(np.int32)
    imgArray=imgArray.astype(np.float32)
    return imgArray

# def getHogimg(img):
#     fd, hog_image = ft.hog(img, orientations=8, pixels_per_cell=(4, 4), 
# cells_per_block=(1, 1),transform_sqrt=None, visualise=True)
#     hog_image.reshape(resize)
#     return hog_image

#修改图片的尺寸
resize = [128,128,1]


def move_file(fp,np,label):
    dnlist = fp.split("\\")
    fname = dnlist[-1]
    childdir = dnlist[-2]
    #如果结果是0，那么图片前面加上的是字母a，其它同理
    if(label==0):
        label = "a"
    elif(label==1):
        label = "b"
    elif(label==2):
        label = "c"
    elif(label==3):
        label = "d"
    elif(label==4):
        label = "e"
    elif(label==5):
        label = "f"
    elif(label==6):
        label = "g"
    elif(label==7):
        label = "h"
    newpath = np+"/"+childdir
    if(not os.path.exists(newpath)):
        os.mkdir(newpath)
    newpath += "/"+str(label)+fname

    shutil.copyfile(fp,newpath)

def get_label(imgp):
    img = Image.open(imgp)
    #这是切割图片中间的区域
    img = img.crop((256,256,768,768))
    #将切割后的图片缩小为128*128
    img.thumbnail((128,128), Image.ANTIALIAS)

    x = np.array(img,np.float32).reshape(resize)

    # x = imgDepth16to8(x)           #大图需要用到本行，小图不需要。

    x = np.expand_dims(x, axis=0)
    
    x /= 255
    
    #预测
    preds = model.predict(x)

    label = np.argmax([preds[0]])

    return label

def runClassFier(path,newpath):
    path += "/*/*.png"
    filep = glob.glob(path)
    total = len(filep)

    for i in range(total):
        imgp = filep[i]
        label = get_label(imgp)
        move_file(imgp,newpath,label)

        yield i+1,total

if __name__=="__main__":
    print("start")
    from keras.utils import plot_model
    plot_model(model, to_file='model.png',show_shapes=True)
