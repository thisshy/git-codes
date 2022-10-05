# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 17:35:26 2021

@author: 11870
"""
import scipy.integrate as si
import numpy as np
from scipy import *
import matplotlib.pyplot as plt
import math
import random


# //////图片的信息标签可以输入中文的代码如下
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams['axes.unicode_minus'] = False  

H0=(67.8*10**-19)/3.086
def f1(x):
    e1=1/np.power((0.692+0.308*np.power(1+x,3)),1/2)
    return e1
def f2(x):
    return si.quad(f1,0,x)[0]
def f3(x):
    e1=1/(np.power((0.692+0.308*np.power(1+x,3)),1/2)*(1+x)**2)
    return e1
def f4(x):
    return si.quad(f1,0,x)[0]
def hf(f,zc,z,m1,m2,e0,f0,lambdag,omega,beta,sigma,phic,S,tc):
    M=m1+m2
    eta=(m1*m2)/(m1+m2)**2
    Mzc=M*(1+zc)*(eta**(3/5))
    Mz=M*(eta**(3/5))
    x=np.power(np.pi*M*f,2/3)
    Ie=np.power(np.pi*M,19/9)*np.power(e0,2)*np.power(f0,19/9)
    betag=(Mz*67.8*np.pi**2/lambdag**2)*f4(z)
    A=67.8*np.power(Mz,5/6)/(np.power(np.pi,2/3)*30**(1/2)*(1+z)*f2(z))
    psi1=(-3*25/(128*768))*np.power(np.pi*Mzc*f,-5/3)*Mzc*(67.8/2)*(np.pi*M*f)**(-8/3)*(1-
     ((np.power((0.692+0.308*np.power(1+zc,3)),1/2))/(1+zc)))
    psi2=2*np.pi*f*tc-phic-(np.pi/4)+(3/128)*np.power(np.pi*Mz*f,-5/3)*(1-(5/84)*(S**2)*(1/omega)*(1/x)-
     (128*betag*x*eta**(2/5)/3)-(2355*Ie*x**(-19/6)/1462)+x*((3715/756)-(55*eta/9))-4*(4*np.pi-beta)*x**(3/2)+
      ((15293365/508032)+(27145*eta/504)+(3085*(eta**2)/72)-10*sigma)*x**2)
    h__real=(np.power(3,1/2)/2)*A*np.power(f,-7/6)*np.cos(psi1+psi2)*f
    h__imag=(np.power(3,1/2)/2)*A*np.power(f,-7/6)*np.sin(psi1+psi2)*f
    h1__real=(np.power(3,1/2)/2)*A*np.power(f,-7/6)*np.cos(psi2)
    h1__imag=(np.power(3,1/2)/2)*A*np.power(f,-7/6)*np.sin(psi2)
    return [h__real,h__imag,h1__real,h1__imag]

# count=[]
# f_v=[]
# h=[]
# h1=[]
# h_real=[]
# h1_real=[]
# sign_1=0
# sign_2=0
# sign_3=0
# sign_4=0
# sign_5=1000
# while sign_1<sign_5:
#     sign_1+=1
#     count.append(sign_1)
    
# # 获取频率的数值模拟值在f_v列表里
# while sign_4<sign_5:
#     f_v.append(random.uniform(0.0005,0.05))   
#     sign_4+=1
# f_v.sort()

# #获取包含额外相位的h(f)
# while sign_2<sign_5:
#     hh=complex(hf(f_v[sign_2],1,1,1.4,1.4,0.9,0,3.1*10**19,7000,0,2,0,1,0)[0],hf(f_v[sign_2],1,0.2,1989*10**30,1989*10**30,0.2,0,3.1*10**19,6944,0,2,0,1,0)[1])
#     h.append(hh)
#     sign_2+=1

# # 不包含额外相位的h(f)
# while sign_3<sign_5:
#     hhh=complex(hf(f_v[sign_3],1,1,1.4,1.4,0.9,0,3.13,7000,0,2,0,1,0)[2],hf(f_v[sign_3],1,0.2,1989*10**30,1989*10**30,0.2,0,3.1*10**19,6944,0,2,0,1,0)[3])
#     h1.append(hhh)
#     sign_3+=1
# # 获取h和h1的实部
# h_real=np.array(h).real
# h1_real=np.array(h1).real
# # 绘图1
# plt.subplot(321)
# plt.scatter(f_v,h_real,s=0.1)
# # plt.plot(h_real)
# plt.xlabel('f',fontsize=15)
# plt.ylabel('振幅',fontsize=15)
# plt.title('考虑加速膨胀')
# # 绘图2
# plt.subplot(322)
# plt.scatter(f_v,h1_real,s=0.1,color='g')
# # plt.plot(h1_real)
# plt.xlabel('f',fontsize=15)
# plt.ylabel('振幅',fontsize=15)
# plt.title('不考虑加速膨胀')
# # 绘图3
# plt.subplot(323)
# # plt.scatter(count,abs(np.fft.ifft(h)), s=0.1,c='#DC143C')	
# plt.plot(np.fft.ifft(h_real))
# plt.title('h的实部的反傅里叶变换')
# # 绘图4
# plt.subplot(325)
# # plt.scatter(count,abs(np.fft.ifft(h1)), s=0.1,c='#DC143C')	
# plt.plot(abs(np.fft.ifft(h)))
# plt.title('h的反傅里叶变换')
# plt.show()
# plt.subplot(324)
# # plt.scatter(count,abs(np.fft.ifft(h1)), s=0.1,c='#DC143C')	
# plt.plot(np.fft.ifft(h1_real),color='g')
# plt.title('h1的实部的反傅里叶变换')
# plt.show()
# plt.subplot(326)
# # plt.scatter(count,abs(np.fft.ifft(h1)), s=0.1,c='#DC143C')	
# plt.plot(np.fft.ifft(h1),color='g')
# plt.title('h1的反傅里叶变换')
# plt.show()







def re_hf(z):
    f=np.arange(10**-5,100,0.1)
    n=len(f)
    h=np.empty([n,1],dtype=complex)
    h1=np.empty([n,1],dtype=complex)
    sign_1=0
    sign_2=0
    zc=z

    m1=1.4*10*30
    m2=1.4*10*30
    e0=0
    f0=0.0
    tc=0
    phic=random.uniform(0,2*np.pi)
    lambdag=1.5
    omega=6944.0
    S=0.0
    beta=random.uniform(0,9.4)
    sigma=random.uniform(0,2.5)
    h_real=hf(f,zc,z,m1,m2,e0,f0,lambdag,omega,beta,sigma,phic,S,tc)[0]
    h_imag=hf(f,zc,z,m1,m2,e0,f0,lambdag,omega,beta,sigma,phic,S,tc)[1]
    h1_real=hf(f,zc,z,m1,m2,e0,f0,lambdag,omega,beta,sigma,phic,S,tc)[2]
    h1_imag=hf(f,zc,z,m1,m2,e0,f0,lambdag,omega,beta,sigma,phic,S,tc)[3]

    while sign_1<n:
        h[sign_1]=complex(h_real[sign_1],h_imag[sign_1])
        sign_1+=1
    while sign_2<n:
        h1[sign_2]=complex(h1_real[sign_2],h1_imag[sign_2])
        sign_2+=1

    return [h_real,h_imag,h1_real,h1_imag,h]
h_real=re_hf(1)[0]
# h_imag=re_hf(1)[1]
h=re_hf(1)[4]
# plt.plot(h_real)
ht=np.fft.ifft(h).real
# plt.plot(ht)
# np.savetxt('h_real.csv',h_real, delimiter = ',')
# np.savetxt('h_imag.csv',h_imag, delimiter = ',')


aa=np.fft.ifft(h)
bb=aa.real
plt.plot(h_real)





