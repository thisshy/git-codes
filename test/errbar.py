# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 16:02:07 2022

@author: W10
"""

import scipy.integrate as si
import numpy as np
from scipy import *
import matplotlib.pyplot as plt
import ht2


H0=(67.8*10**-19)/3.086
z=np.load('z_3009.npy')[0:1000]

# z=np.arange(0,2,0.1)

kong=[]
kong1=[]
XX=[]
for i in z:
    DD=ht2.DL(i)/(3.0842*10**25)
    kong.append(DD)
    XXX=ht2.re_X(i)/H0
    XX.append(XXX)
    kong1.append(sum(XX))
# plt.plot(kong,XX)
# plt.plot(kong,kong1)



 
# a1=np.load('yucezhi1.npy')[1:507].reshape(1,506)
# a2=np.load('yucezhi2.npy')[1:507].reshape(1,506)
# a3=np.load('yucezhi3.npy')[1:507].reshape(1,506)
# a4=np.load('yucezhi4.npy')[1:507].reshape(1,506)
# a5=np.load('yucezhi5.npy')[1:507].reshape(1,506)
# a6=np.load('yucezhi6.npy')[1:507].reshape(1,506)
# a7=np.load('yucezhi7.npy')[1:507].reshape(1,506)

# a8=np.vstack((a1,a2,a3,a4,a5,a6,a7)).reshape(7,506)

# a9=np.max(a8,axis=0)
# a10=np.min(a8,axis=0)
# a11=np.vstack((a9,a10))




####################################################
xi=np.load('xi_6.npy')*pow(10,-20)/H0
n=len(xi)
x_ave=np.sum(xi,axis=0)/n
sigma=pow(np.sum(pow(xi-XX,2),axis=0)/n,1/2).reshape(1000)

# plt.errorbar(kong[0:1700:50],XX[0:1700:50],yerr=sigma[0:1700:50],fmt='o',ecolor='r',color='b',elinewidth=2,capsize=4,ms=3)
plt.plot(kong,XX)
# plt.plot(kong,x_ave)
plt.errorbar(kong[0:1000:40],XX[0:1000:40],yerr=sigma[0:1000:40],fmt='o',ecolor='r',color='b',elinewidth=2,capsize=4,ms=3)
# plt.plot(kong,kong1)