# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 11:59:48 2022

@author: W10
"""
import scipy
from scipy import signal
import scipy.integrate as si
import numpy as np
from scipy import *
import matplotlib.pyplot as plt


ht1=np.load('lvbo_test1.npy')
sn1=np.load('lvbo_test11.npy')
wn=2*0.05/2
b, a = signal.butter(8,wn,'highpass')
filtedData = signal.filtfilt(b, a,ht1)  #data为要过滤的信号
# plt.plot(sn1)
plt.plot(ht1)
plt.plot(filtedData)


