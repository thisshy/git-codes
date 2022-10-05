# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 16:18:15 2022

@author: W10
"""

import scipy.integrate as si
import numpy as np
from scipy import *
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler



##############################################降低数据分辨率
# a=np.load('516.npy',allow_pickle=True)

def re_data(a,data_size,n=448052):
    # a=np.delete(a,0,axis=0)
    sign1=-100
    sign2=0
    kong1=[]
    while sign1<n:
        sign1+=100
        if sign1<n:
            kong1.append(a[:,sign1])
        if sign1>n-1:
            kong1.append(a[:,-1])
    
    e=np.zeros((data_size,4482))
    while sign2<4482:
        e[:,sign2]=kong1[sign2]
        sign2+=1
    return e


# data3_1=re_data(a,114)
##########################################合并数据

a0=np.load('100_3009.npy',allow_pickle=True)
a1=np.load('200_3009.npy',allow_pickle=True)
a2=np.load('300_3009.npy',allow_pickle=True)
a3=np.load('400_3009.npy',allow_pickle=True)
a4=np.load('500_3009.npy',allow_pickle=True)
a5=np.load('600_3009.npy',allow_pickle=True)
a6=np.load('700_3009.npy',allow_pickle=True)
a7=np.load('800_3009.npy',allow_pickle=True)
a8=np.load('900_3009.npy',allow_pickle=True)
a9=np.load('1000_3009.npy',allow_pickle=True)
# a10=np.load('1100_3009.npy')
# a11=np.load('1200_3009.npy')
# a12=np.load('1300_3009.npy')
# a13=np.load('1400_3009.npy')
# a14=np.load('1500_3009.npy')
# a15=np.load('1600_3009.npy')
# a16=np.load('1700_3009.npy')
# a17=np.load('1800_3009.npy')
# a18=np.load('1900_3009.npy')
# a19=np.load('2000_3009.npy')
# a20=np.load('2100_3009.npy')
# a21=np.load('2200_3009.npy')
# a22=np.load('2300_3009.npy')
# a23=np.load('2400_3009.npy')
# a24=np.load('2500_3009.npy')
# a25=np.load('2600_3009.npy')
# a26=np.load('2700_3009.npy')
# a27=np.load('2800_3009.npy')

# ,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,a25,a26,a27

final=np.vstack((a0,a1,a2,a3,a4,a5,a6,a7,a8,a9))



l1=np.shape(final)[1]
l2=np.shape(final)[0]
def zhuanzhi(l1,l2,final):
    n=0
    n1=0
    kong=np.zeros((l1,l2))
    kong1=[]
    while n<l2:
        kong1.append(final[n,:])
        n+=1
    while n1<l2:
        kong[:,n1]=kong1[n1]
        n1+=1
    return kong

final1=zhuanzhi(l1, l2, final)





