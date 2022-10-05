# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 20:19:21 2022

@author: W10
"""

import scipy.integrate as si
import numpy as np
import matplotlib.pyplot as plt
import math
import sf
import scipy.io as sio




def re_sn(f,N):
    w=sf.re_sf(f)
    fs=2*np.max(f)
    L=len(f)
    w1=w
    w1=w1.tolist()
    w1=w1[::-1]
    w1=np.array(w1)
    w2=np.hstack((w,w1))
    Ax=np.sqrt(w*N)
    Ax1=np.sqrt(w1*N)
    sign_1=0
    sign_2=0
    sign_3=0
    sign_4=0
    kong_1=[]
    while sign_1<L:
        kong_1.append(np.random.uniform(0,2*np.pi))
        sign_1+=1
    kong_11=kong_1
    kong_11=kong_11[::-1]
    def re_fushu(theta):
        expr1=np.cos(theta)
        expr2=np.sin(theta)
        return [expr1,expr2]
    
    kong_2=[]
    while sign_2<L:
        kong_2.append(Ax[sign_2]*complex(re_fushu(kong_1[sign_2])[0],re_fushu(kong_1[sign_2])[1]))
        sign_2+=1
    kong_3=[]
    while sign_3<L:
        kong_3.append(Ax1[sign_3]*complex(re_fushu(kong_11[sign_3])[0],re_fushu(kong_11[sign_3])[1]))
        sign_3+=1
    kong_2=np.array(kong_2)
    kong_3=np.array(kong_3)
    f_two=np.hstack((kong_2,kong_3))
    ifft_f_two=np.fft.ifft(f_two,N).real
    return [ifft_f_two,w]


ht=np.load('a.npy')

fmin=0.4
fmax=1
N=20000000
M=int((N/2)+1)
f=np.linspace(fmin,fmax,M)
dt=pow(2*fmax,-1)

tmax=(N-1)*dt
t=np.linspace(0,tmax,N)
noise=re_sn(f,N)[0]
b=re_sn(f,N)[1]
# plt.subplot(211)
# plt.plot(t,a)
# plt.subplot(212)
# plt.loglog(f,np.sqrt(b))


plt.plot(t,noise)

plt.plot(t,ht)






