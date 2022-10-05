# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 20:30:06 2021

@author: 11870
"""

import scipy.integrate as si
import numpy as np
from scipy import *
import numpy as np	
import matplotlib.pyplot as plt
import scipy.stats as st
import math
import random
import collections
import latexify
# //////图片的信息标签可以输入中文的代码如下
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams['axes.unicode_minus'] = False  





# x=np.linspace(0,7,1400)

# 从时域信号变换到频域，傅里叶变换，并且计算出相应的频率对应的振幅
Fs = 5000;                 # 采样率 (决定了频率区间)
Ts = 1.0/Fs;                # 采样区间
t = np.arange(0,3,Ts)       # 时间矢量，这里Ts也是步长
#range返回从0到1构成的list，而arange返回一个array对象

fs = 2;                    # frequency of the signal信号频率
y = np.sin(200*t**4)+np.sin(600*t)+((t**4)*pow(1-t**2,2)+t*np.exp(2+t))/500

n = len(t)                  # 信号长度
k = np.arange(n)            #采样点数的等差数列k
T = n/Fs                    #共有多少个CAIYANG周期T
frq = k/T                   # two sides frequency range两侧频率范围
frq1 = frq[range(int(n/2))] # #由于对称性，取一半区间

Y= np.fft.fft(y)          # 未归一化
# Y = np.fft.fft(y)/n         #  归一化
# Y1 = Y[range(int(n/2))]


y1=np.sin(200*t**4)+np.sin(600*t)
y2=((t**4)*pow(1-t**2,2))/500
y3=(t*np.exp(2+t))/500
Y1=np.fft.fft(y1)
Y2=np.fft.fft(y2)
Y3=np.fft.fft(y3)
Y4=Y1+Y2+Y3
Y5=np.fft.ifft(Y4)

# # /////////////////////////////////////////////////////////////////////////////////
# fig, ax = plt.subplots(9, 1)

# ax[0].plot(t,y)
# ax[0].set_xlabel('Time')
# ax[0].set_ylabel('Amplitude振幅')
# ax[0].set_title('sin(200t)+sin(600t)')
# ax[1].plot(frq,abs(YY),'r') #绘制频谱
# ax[1].set_xlabel('Freq (Hz)')
# ax[1].set_ylabel('|Y(freq)|')

# ax[2].plot(frq,abs(Y),'g')  # plotting the spectrum
# ax[2].set_xlabel('Freq (Hz)')
# ax[2].set_ylabel('|Y(freq)|')

# ax[1].plot(frq1,abs(YY[range(int(n/2))]),'b') # plotting the spectrum
# ax[1].set_xlabel('Freq (Hz)')
# ax[1].set_ylabel('|Y(freq)|')

# ax[4].plot(t,np.fft.ifft(YY),'b') # plotting the spectrum
# ax[4].set_xlabel('Freq (Hz)')
# ax[4].set_ylabel('|Y(freq)|')



# ax[2].plot(t,y1,'b') # plotting the spectrum
# ax[2].set_title('sin(200t)')
# ax[3].plot(frq1,abs(Y1[range(int(n/2))]),'b') # plotting the spectrum
# ax[4].plot(t,y2,'b') # plotting the spectrum
# ax[4].set_title('sin(600y)')
# ax[5].plot(frq1,abs(Y2[range(int(n/2))]),'b') # plotting the spectrum

# ax[6].plot(t,y3,'b') # plotting the spectrum
# ax[6].set_title('sin(200t)')
# ax[7].plot(frq1,abs(Y3[range(int(n/2))]),'b') # plotting the spectrum

# ax[8].plot(t,Y5,'g') # plotting the spectrum
# ax[8].set_title('频域相加再反傅里叶变换回来')
# # ax[9].plot(frq1,abs(Y4[range(int(n/2))]))
# plt.show()
# /////////////////////////////////////////////////////////////////////////////////




# /////////////////////////////////////////////////////////////////////////////////

# plt.subplot(5,2,1)
# plt.plot(t,y,color='#EE82EE')
# plt.title('sin(200t^4)+sin(600t)+(t^4*(1-t^2)^2+t*e^(2+t))/500')

# plt.subplot(5,2,2)
# plt.plot(frq1,abs(Y[range(int(n/2))]),color='#EE82EE')

# plt.subplot(5,2,3)
# plt.plot(t,y1,color='#00FFFF')
# plt.title('sin(200t^4)+sin(600t)')


# plt.subplot(5,2,4)
# plt.plot(frq1,abs(Y1[range(int(n/2))]),color='#00FFFF')

# plt.subplot(5,2,5)
# plt.plot(t,y2,color='b')
# plt.title('(t^4*(1-t^2)^2)/500')


# plt.subplot(5,2,6)
# plt.plot(frq1,abs(Y2[range(int(n/2))]),color='b')

# plt.subplot(5,2,7)
# plt.plot(t,y3,color='#8A2BE2')
# plt.title('(t*e^(2+t))/500)')


# plt.subplot(5,2,8)
# plt.plot(frq1,abs(Y3[range(int(n/2))]),color='#8A2BE2')




# plt.subplot(5,2,9)
# plt.plot(frq1,abs(Y4[range(int(n/2))]),color='g')
# plt.title('频域相加')
# plt.subplot(5,2,10)
# plt.plot(t,Y5,color='g')
# plt.title('频域相加后的反傅里叶变换')

# plt.show
# /////////////////////////////////////////////////////////////////////////////////

# abs_Y=abs(Y[range(int(n/2))])
# abs_Y1=abs(Y1[range(int(n/2))])
# abs_Y2=abs(Y2[range(int(n/2))])
# abs_Y3=abs(Y3[range(int(n/2))])
# abs_Y4=abs(Y4[range(int(n/2))])

# aa=np.empty([7500,5],dtype=float)
# b=0
# while b<7500:
#     aa[b,0]=abs_Y[b]
#     b+=1
# b=0
# while b<7500:
#     aa[b,1]=abs_Y1[b]
#     b+=1
# b=0
# while b<7500:
#     aa[b,2]=abs_Y2[b]
#     b+=1
# b=0
# while b<7500:
#     aa[b,3]=abs_Y3[b]
#     b+=1
# b=0
# while b<7500:
#     aa[b,4]=abs_Y4[b]
#     b+=1




