# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 13:14:51 2022

@author: 11870
"""
import scipy.integrate as si
import numpy as np
import matplotlib.pyplot as plt
import math

# import hf1
# # X=hf1.re_hf(1)[0].reshape(1,-1)
# # e=hf1.re_hf(1)[0]
# # 读取数据
# import pandas as pd
# X = pd.read_csv("ht_real.csv")

# X=np.array(X).reshape(9999,1)
# b=X
# X -= np.mean(X) # 减去均值，使得以0为中心
# X /= np.std(X) # 归一化
# X -= np.mean(X) # 减去均值，使得以0为中心
# cov = np.dot(X.T, X) / X.shape[0] #计算协方差矩阵
# U,S,V = np.linalg.svd(cov) #矩阵的奇异值分解
# Xrot = np.dot(X, U) 
# Xwhite = Xrot / np.sqrt(S + 1e-5) #加上1e-5是为了防止出现分母为0的异常

# c=np.array([x for y in Xwhite for x in y])
# a=np.arange(9999)
# plt.plot(a,c)

