# -*- coding: utf-8 -*-
"""
Created on Mon May  2 14:23:39 2022

@author: W10
"""

import numpy as np
import matplotlib.pyplot as plt
import ht2
import sf
import scipy.integrate as si

# ///////////////////////////////////时域计算信噪比

# ht=np.load('ceshiht.npy')
# sn=np.load('ceshisn.npy')
# a=pow(ht,2)
# snr=10*np.log10(sum(pow(ht,2))/sum(pow(sn,2)))

# ifft_ht=np.fft.fft(ht)
# ifft_sn=np.fft.fft(sn)

# 、、///////////////////////////////////////////////


f=np.arange(10**-3,1,10**-6)

H0=(70*10**-19)/3.086
G=6.67*10**-11
c=2.99792458*pow(10,8)
m1=1.4*1.989*(10**30)*pow(10,3)
m2=1.4*1.989*(10**30)*pow(10,3)
M=m1+m2
cM=pow(m1*m2,3/5)/pow(M,1/5)


def H(z):
    e1=H0*(0.7+0.3*pow(1+z, 3))**0.5
    return e1
def f1(z):
    return 1/H(z)
def f2(z):
    return 1/(H(z)*((1+z)**2))
def f3(z):
    return si.quad(f1,0,z)[0]
def f4(z):
    return si.quad(f2,0,z)[0]
def D(z):
    return (1+z)*f4(z)*c
def DL(z):
    return (1+z)*f3(z)*c

cc=DL(0.5)

A=(pow(3,1/2)/2)*(1/(pow(30,1/2)*pow(np.pi,2/3)))*pow(G*cM/pow(c,3),5/6)/(DL(0.5)/c)
e1=pow(f,-7/3)/pow(sf.re_sf(f),1)
snr=2*A*pow(si.simps(e1,f),1/2)




z=0.524
dl=DL(z)/(3.0842*10**25)


