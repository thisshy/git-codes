# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 21:03:54 2021

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

# a=np.empty([5,5],dtype=float)
# b=np.arange(0,25,1,dtype=float).reshape((5,5))
# c=np.c_[a,b]
# print(np.max(b,axis=1))

# a=np.array([2,1,3,4,5])
# b=np.array([1,2,4,2,7])
# c=np.c_[a,b]
# print(np.min(c,axis=1))


# f=np.arange(10**-5,10**2,0.0001)
def S_decigo(f):
    expr1=1.3*pow(f,-4)*pow(10,-52)
    expr2=4.6*pow(10,-49)
    expr3=2.0*pow(f,2)*pow(10,-49)
    return expr1+expr2+expr3
def S_gal(f):
    expr1=2.1*pow(10,-45)*pow(f,-7/3)
    return expr1
def S_exgal(f):
    expr1=4.2*pow(10,-47)*pow(f,-7/3)
    return expr1
def S_ns(f):
    expr1=1.3*pow(10,-48)*pow(f,-7/3)
    return expr1
def S_sum(f):
    K=4.5
    dn_df=2*pow(10,-3)*pow(f,-11/3)
    F=np.exp(-2*pow(f/0.05,2))
    T=pow(10,7)*3.1536
    expr0=S_exgal(f)*F+0.01*S_ns(f)
    expr1=(S_decigo(f))/(np.exp(-K*dn_df*(1/T)))+expr0
    expr2=S_decigo(f)+S_gal(f)*F+expr0
    expr3=np.c_[expr1,expr2]
    expr4=np.min(expr3,axis=1)
    return expr4

# sf=S_sum(f)
# plt.subplot(2,1,1)
# plt.plot(f,pow(sf,1/2))
# plt.title('Decigo--Sn(f)')
# plt.xlabel('HZ')
# plt.ylabel('Sn^1/2')
# plt.subplot(2,1,2)
# sf_ifft=np.fft.ifft(sf)
# plt.plot(sf_ifft)
# plt.title('时域')
# plt.xlabel('t')
# plt.ylabel('振幅')

def re_sf(f):
    return S_sum(f)


f=np.arange(0.05,1,10**-6)

a=re_sf(f)

