# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 15:04:39 2021

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


# //////图片的信息标签可以输入中文的代码如下
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams['axes.unicode_minus'] = False

H0=(67.8*10**-19)/3.086
G=6.67*10**-11
c=3.0*10**8

def H(x):
    e1=H0*(0.692+0.308*pow(1+x, 3))**0.5
    return e1
def f1(x):
    return 1/H(x)
def f2(x):
    return 1/(H(x)*(1+x)**2)
def f3(x):
    return si.quad(f1,0,x)[0]
def f4(x):
    return si.quad(f2,0,x)[0]
def D(x):
    return (1+x)*f4(x)*c
def DL(x):
    return (1+x)*f3(x)*c
def hf(f,zc,z,m1,m2,e0,f0,tc,phic,lambdag,omegabd,S,beta,sigma):
    M=m1+m2
    eta=m1*m2/(M**2)
    Mzc=M*(1+zc)*pow(eta, 3/5)
    Mz=M*pow(eta, 3/5)
    xzc=(np.pi*Mzc*f)**(2/3)*pow(c,-2)*pow(G,2/3)
    xz=(np.pi*Mz*f)**(2/3)*pow(c,-2)*pow(G,2/3)
    Ie=pow(np.pi*M*f0,19/9)*e0**2*pow(G,19/9)*pow(c,-19/3)
    omega=1/omegabd
    betag=(D(z)*Mz*np.pi**2)/((1+z)*lambdag**2)*pow(G,1)*pow(c,-2)
    PSIN=(3/128)*pow(np.pi*Mzc*f,-5/3)
    X=(H0/2)*(1-(H(zc)/((1+zc)*H0)))
    A=(pow(3,0.5)/2)*pow(f,-7/6)*(pow(Mz,5/6)/(pow(30,0.5)*pow(np.pi,2/3)*DL(z)))*pow(G,5/6)*pow(c,-3/2)
    PSIacc=-PSIN*X*(25/768)*Mzc*pow(xzc,-4)*pow(G, -10/3)*pow(c, 10)
    expr1=1
    expr2=(-5/84)*pow(S,2)*omega*pow(xz,-1)
    expr3=(-128/3)*betag*pow(eta,2/5)*xz
    expr4=(-2355/1426)*Ie*pow(xz,-19/6)
    expr5=(3715/756+(55*eta/9))*xz
    expr6=-4*(4*np.pi-beta)*pow(xz,3/2)
    expr7=pow(xz,2)*(15293365/508032+27145*eta/504+3085*pow(eta,2)/72-10*sigma)
    PSI=2*np.pi*f*tc-phic-(np.pi/4)+(3/128)*pow(np.pi*Mz*f,-5/3)*(expr1+expr2+expr3+expr4+expr5+expr6+expr7)*pow(G,-5/3)*pow(c,5)
    h_real=A*np.cos(PSIacc+PSI)
    h_imag=A*np.sin(PSIacc+PSI)
    h1_real=A*np.cos(PSI)
    h1_imag=A*np.sin(PSI)
    # print(h_real)
    return [h_real,h_imag,h1_real,h1_imag,X]
def X(zc):
    return (H0/2)*(1-(H(zc)/((1+zc)*H0)))


# f=np.arange(10**-5,1,0.001)
# n=len(f)
# h=np.empty([n,1],dtype=complex)
# h1=np.empty([n,1],dtype=complex)
# sign_1=0
# sign_2=0
# sign_3=0
# sign_4=0
# zc=z=1.0 

# m1=1.4*1.989*10**30
# m2=1.4*1.989*10**30
# e0=0.5
# f0=0.0
# tc=0
# phic=random.uniform(0,2*np.pi)
# lambdag=3.1*10**7
# omegabd=6944.0
# S=0.0
# beta=random.uniform(0,9.4)
# sigma=random.uniform(0,2.5)
    


# h_real=hf(f,zc,z,m1,m2,e0,f0,tc,phic,lambdag,omegabd,S,beta,sigma)[0]
# h_imag=hf(f,zc,z,m1,m2,e0,f0,tc,phic,lambdag,omegabd,S,beta,sigma)[1]
# h1_real=hf(f,zc,z,m1,m2,e0,f0,tc,phic,lambdag,omegabd,S,beta,sigma)[2]
# h1_imag=hf(f,zc,z,m1,m2,e0,f0,tc,phic,lambdag,omegabd,S,beta,sigma)[3]
# while sign_1<n:
#     h[sign_1]=complex(h_real[sign_1],h_imag[sign_1])
#     sign_1+=1
# while sign_2<n:
#     h1[sign_2]=complex(h1_real[sign_2],h1_imag[sign_2])
#     sign_2+=1
    

# ifft_h=np.fft.ifft(h)
# ifft_h1=np.fft.ifft(h1)

# ifft_h=ifft_h.tolist()
# ifft_h.reverse()
# ifft_h1=ifft_h1.tolist()
# ifft_h1.reverse()


# # 绘图1
# plt.subplot(221)
# plt.plot(f,h,color='b')
# plt.xlabel('f',fontsize=15)
# plt.ylabel('振幅',fontsize=15)
# plt.title('考虑加速膨胀')
# # 绘图2
# plt.subplot(222)
# plt.plot(f,abs(h1),color='g')
# plt.xlabel('f',fontsize=15)
# plt.ylabel('振幅',fontsize=15)
# plt.title('不考虑加速膨胀')

# # 绘图3
# plt.subplot(223)
# # plt.scatter(count,abs(np.fft.ifft(h1)), s=0.1,c='#DC143C')	
# plt.plot(ifft_h,color='b')
# plt.title('h的反傅里叶变换')

# # 绘图4
# plt.subplot(224)
# # plt.scatter(count,abs(np.fft.ifft(h1)), s=0.1,c='#DC143C')	
# plt.plot(ifft_h1,color='g')
# plt.title('h1的反傅里叶变换')



# aa=abs(h)
# b=abs(h1)
# print(abs(h[1]),abs(h1[1]))


# aa=np.arange(0,2,0.01)
# bb=np.empty(len(aa),dtype=float)
# ii=0
# for i in aa:
#     bb[ii]=DL(i)
#     ii+=1
# cc=X(aa)
# fig,ax1 = plt.subplots()
# ax2 = ax1.twinx()           #给ax1做镜像处理变为ax2
# ax1.plot(cc,aa,'b-',linewidth=1)#主坐标轴画降水过程图
# ax2.plot(cc,bb,'g-',linewidth=0.5)#次坐标轴画径流过程图
# ax1.set_xlabel('X')    #设置x轴标题
# ax1.set_ylabel('z',color='b')   #设置主轴标题
# ax2.set_ylabel('DL',color = 'g')   #设置次轴标题
# # ax1.set_title('降水、径流过程图',fontsize=20)#设置图表名称
# plt.show()#显示图片

  # 调用函数返回hf值
  
# def re_hf(z):
#     f=np.arange(10**-5,10**2,0.1)
#     n=len(f)
#     h=np.empty([n,1],dtype=complex)
#     h1=np.empty([n,1],dtype=complex)
#     sign_1=0
#     sign_2=0
#     zc=z

#     m1=1.4*1.989*10**30
#     m2=1.4*1.989*10**30
#     e0=0.5
#     f0=0.0
#     tc=0
#     phic=random.uniform(0,2*np.pi)
#     lambdag=3.1*10**19
#     omegabd=6944.0
#     S=0.0
#     beta=random.uniform(0,9.4)
#     sigma=random.uniform(0,2.5)
#     h_real=hf(f,zc,z,m1,m2,e0,f0,tc,phic,lambdag,omegabd,S,beta,sigma)[0]
#     h_imag=hf(f,zc,z,m1,m2,e0,f0,tc,phic,lambdag,omegabd,S,beta,sigma)[1]
#     h1_real=hf(f,zc,z,m1,m2,e0,f0,tc,phic,lambdag,omegabd,S,beta,sigma)[2]
#     h1_imag=hf(f,zc,z,m1,m2,e0,f0,tc,phic,lambdag,omegabd,S,beta,sigma)[3]
#     X=hf(f,zc,z,m1,m2,e0,f0,tc,phic,lambdag,omegabd,S,beta,sigma)[4]/H0
#     while sign_1<n:
#         h[sign_1]=complex(h_real[sign_1],h_imag[sign_1])
#         sign_1+=1
#     while sign_2<n:
#         h1[sign_2]=complex(h1_real[sign_2],h1_imag[sign_2])
#         sign_2+=1
#     abs_h=abs(h)
#     abs_h1=abs(h1)
#     return [abs_h,abs_h1,X]




# def re_hf(z):
#     f=np.arange(10**-5,10**2,0.1)
#     n=len(f)
#     h=np.empty([n,1],dtype=complex)
#     h1=np.empty([n,1],dtype=complex)
#     sign_1=0
#     sign_2=0
#     zc=z

#     m1=1.4*1.989*10**30
#     m2=1.4*1.989*10**30
#     e0=0.5
#     f0=0.0
#     tc=0
#     phic=random.uniform(0,2*np.pi)
#     lambdag=3.1*10**19
#     omegabd=6944.0
#     S=0.0
#     beta=random.uniform(0,9.4)
#     sigma=random.uniform(0,2.5)
#     h_real=hf(f,zc,z,m1,m2,e0,f0,tc,phic,lambdag,omegabd,S,beta,sigma)[0]
#     h_imag=hf(f,zc,z,m1,m2,e0,f0,tc,phic,lambdag,omegabd,S,beta,sigma)[1]
#     h1_real=hf(f,zc,z,m1,m2,e0,f0,tc,phic,lambdag,omegabd,S,beta,sigma)[2]
#     h1_imag=hf(f,zc,z,m1,m2,e0,f0,tc,phic,lambdag,omegabd,S,beta,sigma)[3]
#     X=hf(f,zc,z,m1,m2,e0,f0,tc,phic,lambdag,omegabd,S,beta,sigma)[4]/H0
#     while sign_1<n:
#         h[sign_1]=complex(h_real[sign_1],h_imag[sign_1])
#         sign_1+=1
#     while sign_2<n:
#         h1[sign_2]=complex(h1_real[sign_2],h1_imag[sign_2])
#         sign_2+=1

#     return [h,h1,X]



def re_hf(z):
    f=np.arange(10**-5,2,0.0002)
    n=len(f)
    h=np.empty([n,1],dtype=complex)
    h1=np.empty([n,1],dtype=complex)
    sign_1=0
    sign_2=0
    zc=z

    m1=1.4*1.989*10**35
    m2=1.4*1.989*10**35
    e0=0.5
    f0=0.0
    tc=0
    phic=random.uniform(0,2*np.pi)
    lambdag=3.1*10**19
    omegabd=6944.0
    S=0.0
    beta=random.uniform(0,9.4)
    sigma=random.uniform(0,2.5)
    h_real=hf(f,zc,z,m1,m2,e0,f0,tc,phic,lambdag,omegabd,S,beta,sigma)[0]
    h_imag=hf(f,zc,z,m1,m2,e0,f0,tc,phic,lambdag,omegabd,S,beta,sigma)[1]
    h1_real=hf(f,zc,z,m1,m2,e0,f0,tc,phic,lambdag,omegabd,S,beta,sigma)[2]
    h1_imag=hf(f,zc,z,m1,m2,e0,f0,tc,phic,lambdag,omegabd,S,beta,sigma)[3]
    X=hf(f,zc,z,m1,m2,e0,f0,tc,phic,lambdag,omegabd,S,beta,sigma)[4]

    return [h_real,h_imag,X]


# ############################################DL—XH图
z=np.arange(0,2,0.1)
aaa=len(z)
kong=[]
kong1=[]
XX=[]
for i in z:
    DD=DL(i)/(3.0842*10**25)
    kong.append(DD)
    XXX=X(i)/H0
    XX.append(XXX)
    kong1.append(sum(XX))
plt.plot(kong,XX)
plt.plot(kong,kong1)
