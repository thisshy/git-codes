
# from skimage import io
# from skimage import feature as ft
import glob
from PIL import Image,ImageDraw
import numpy as np
import os

# path = "E:\\machine\\ML6切后N类\\数据\\原始数据\\全黑有星星\\20131004022937_LQu408_7_8658_60000_B1_G3.png"
# path3 = "C:\\Users\\root_weiyi\\Desktop\\e20140204184300_LQu408_7_8658_60000_B1_G3.png"

# img = Image.open(path3)
# draw = ImageDraw.Draw(img)
# draw.line([(212,212),(812,212)],fill=128)
# draw.line([(212,212),(212,812)],fill=128)
# draw.line([(812,212),(812,812)],fill=128)
# draw.line([(212,812),(812,812)],fill=128)

# draw.line([(256,256),(768,256)],fill=128)
# draw.line([(256,256),(256,768)],fill=128)
# draw.line([(768,256),(768,768)],fill=128)
# draw.line([(256,768),(768,768)],fill=128)

# img.show()

# img.save("切割范围2.png")

#切割图像后保存的路径，要先手动新建好文件夹
toP = "./切割后图片2/训练集/"

def cr(path):
    img = Image.open(path)
    img = img.crop((256,256,768,768))
    imgName = path.split("\\")[-1]
    filePath = toP+path.split("\\")[-2]
    if(not os.path.exists(filePath)):
        os.mkdir(filePath)
    imgName = filePath+"/"+imgName
    print(imgName)
    img.save(imgName)


#将/数据2/训练集/路径下所有子文件夹内png图像切割为512*512
for p in glob.glob("./训练集/*/*.png"):
    cr(p)