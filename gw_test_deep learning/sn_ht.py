# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 22:02:50 2022

@author: W10
"""

import scipy.integrate as si
import numpy as np
from scipy import *
import matplotlib.pyplot as plt
import ht2
import pro_z
import sn
from scipy import signal
# z=pro_z.re_z(3000,23)[0]



zzz=np.load('z_3009.npy')
z=np.load('z_3009.npy')[0:1]





# def re_data_new(a,data_size,n=448051):
#     # a=np.delete(a,0,axis=0)
#     sign1=-100
#     sign2=0
#     kong1=[]
#     while sign1<n:
#         sign1+=100
#         if sign1<n:
#             kong1.append(a[:,sign1])
#         if sign1>n-1:
#             kong1.append(a[:,-1])
    
#     e=np.zeros((data_size,4482))
#     while sign2<4482:
#         e[:,sign2]=kong1[sign2]
#         sign2+=1
#     return e





# def re_data(f,f1,z):
#     n=len(z)
#     sign1=0
#     sign2=0
#     kong1=[]
#     kong3=[] 
#     kong4=[]
#     while sign1<n:
#         ht=ht2.re_ht(f,z[sign1])
#         d=len(ht)
#         sn1=sn.re_sn(f1,d)
#         htsn=ht+sn1
#         htsn1=htsn[int(0.7*d):]
#         htsn1=htsn1*pow(10,21)
#         kong1.append(htsn1)
#         # kong3.append(ht)
#         # kong4.append(sn1)
#         sign1+=1
#     return [kong1]


def re_x(z):
    sign1=0
    kong1=[]
    n=len(z)
    while sign1<n:
        X=ht2.re_X(z[sign1])*10**20
        kong1.append(X)
        sign1+=1
    return kong1




def re_ht1(f,z,n=448051):
    n0=len(z)
    sign0=0
    kong0=[]
    while sign0<n0:
        ht=ht2.re_ht(f,z[sign0])
        d=len(ht)
        ht1=ht[int(0.7*d):]
        kong0.append(ht1)
        sign0+=1
    a=np.array(kong0)
# 降低分辨率
    sign1=-100
    sign2=0
    kong1=[]
    while sign1<n:
        sign1+=100
        if sign1<n:
            kong1.append(a[:,sign1])
        if sign1>n-1:
            kong1.append(a[:,-1])
    
    e=np.zeros((n0,4482))
    while sign2<4482:
        e[:,sign2]=kong1[sign2]
        sign2+=1
    wn=2*0.05/2
    b,a = signal.butter(8,wn,'highpass')
    filtedData = signal.filtfilt(b,a,e,axis=1)
        
    return filtedData



def re_d(f):
    e=ht2.re_ht(f,1)
    return len(e)




def re_sn1(f,d,z,n=448051):
    sign0=0
    n0=len(z)
    kong0=[]
    while sign0<n0:
        sn1=sn.re_sn(f,d)
        sn2=sn1[int(0.7*d):]
        kong0.append(sn2)
        sign0+=1
    a=np.array(kong0)
# 降低分辨率
    sign1=-100
    sign2=0
    kong1=[]
    while sign1<n:
        sign1+=100
        if sign1<n:
            kong1.append(a[:,sign1])
        if sign1>n-1:
            kong1.append(a[:,-1])
    
    e=np.zeros((n0,4482))
    while sign2<4482:
        e[:,sign2]=kong1[sign2]
        sign2+=1
        
    return e


def re_data(ht,sn1):
    e=ht+sn1
    return e*pow(10,22)
    


f=np.arange(10**-3,100,10**-4)
f1=np.arange(10**-3,100,10**-4)



n=len(z)
X=np.array(re_x(z)).reshape(n ,1)
ht1=re_ht1(f,z)
d=re_d(f)
sn1=re_sn1(f1,d,z)
sn_ht=re_data(ht1,sn1)
data=np.hstack((sn_ht,X))

sn_ht=sn_ht.reshape(-1)
sn1=sn1.reshape(-1)
ht1=ht1.reshape(-1)
plt.plot(sn1)
plt.plot(ht1)







