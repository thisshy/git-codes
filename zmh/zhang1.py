# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 18:04:53 2022

@author: 11870
"""

import scipy
import numpy as np
import matplotlib.pyplot as plt

sv=np.load('sv.npy')
sv=sv.tolist()
n=np.linspace(1,500,500,dtype=int)
sn=[]
for i in n:
    sn.append(np.array(np.random.randn(1000)*5-40))

data=[]
label=[]
for i in n:
    if i%2!=0:
        data.append(np.array(sn[i-1]))
        label.append(0)
    if i%2==0:
        data.append(np.array(sv[i-1])+sn[i-1])
        label.append(1)

data=np.array(data)
label=np.array(label).reshape(500,1)

data1=np.hstack((data,label))