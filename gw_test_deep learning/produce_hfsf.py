# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 15:24:42 2021

@author: 11870
"""
import sf
import hf2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# a1=sf.re_sf().reshape(1,1000)

# z=np.arange(1**-7,2,0.001)
# aaa=len(z)
# kong=[]
# XX=[]
# for i in z:
#     a2=hf2.re_hf(i)[0].reshape(1,1000)

#     X=hf2.re_hf(i)[2]
#     a3=a1+a2
#     kong.append(a3)
#     XX.append(X)

# aa=np.array(kong)
# hf_sf=aa[:,-1]
# X=np.array(XX).reshape(len(z),1)/max(XX)

# # hf_sf_X=np.hstack((hf_sf,X))
# mean = hf_sf.mean(axis=0) #axis=0表示每一列的数据
# hf_sf -= mean
# std = hf_sf.std(axis=0)
# hf_sf /= std

# mean = hf_sf.mean(axis=0) #axis=0表示每一列的数据
# hf_sf -= mean
# std = hf_sf.std(axis=0)
# hf_sf /= std

# z1=np.arange(2,3,0.01)
# kong1=[]
# XX1=[]
# for i in z1:
#     a22=hf2.re_hf(i)[0].reshape(1,1000)

#     X1=hf2.re_hf(i)[2]
#     a33=a1+a22
#     kong1.append(a33)
#     XX1.append(X1)

# aaa=np.array(kong1)
# hf_sf1=aaa[:,-1]
# X1=np.array(XX1).reshape(len(z1),1)/min(XX1)

# mean1 = hf_sf1.mean(axis=0) #axis=0表示每一列的数据
# hf_sf1 -= mean
# std = hf_sf1.std(axis=0)
# hf_sf1 /= std

# # hf_sf_X1=np.hstack((hf_sf1,X1))
# np.savetxt('1.csv', hf_sf, delimiter = ',')
# np.savetxt('2.csv', hf_sf1, delimiter = ',')
# np.savetxt('3.csv', X, delimiter = ',')
# np.savetxt('4.csv', X1, delimiter = ',')





# a1=sf.re_sf().reshape(1,1000)

# z=np.arange(1**-7,2,0.001)
# aaa=len(z)
# kong=[]
# XX=[]
# for i in z:
#     a2=hf2.re_hf(i)[0].reshape(1,1000)

#     X=hf2.re_hf(i)[2]
#     a3=a1+a2
#     kong.append(a3)
#     XX.append(X)

# aa=np.array(kong)
# hf_sf=aa[:,-1]
# X=np.array(XX).reshape(len(z),1)


# mean = hf_sf.mean(axis=0) #axis=0表示每一列的数据
# hf_sf -= mean
# std = hf_sf.std(axis=0)
# hf_sf /= std
# hf_sf_X=np.hstack((hf_sf,X))
# np.savetxt('data.csv', hf_sf_X, delimiter = ',')




# z1=np.arange(2,3,0.01)
# kong1=[]
# XX1=[]
# for i in z1:
#     a22=hf2.re_hf(i)[0].reshape(1,1000)

#     X1=hf2.re_hf(i)[2]
#     a33=a1+a22
#     kong1.append(a33)
#     XX1.append(X1)

# aaa=np.array(kong1)
# hf_sf1=aaa[:,-1]
# X1=np.array(XX1).reshape(len(z1),1)

# mean1 = hf_sf1.mean(axis=0) #axis=0表示每一列的数据
# hf_sf1 -= mean1
# std1 = hf_sf1.std(axis=0)
# hf_sf1 /= std1

# hf_sf_X1=np.hstack((hf_sf1,X1))

# np.savetxt('data1.csv', hf_sf_X1, delimiter = ',')

a=hf2.re_hf(0.5)[0]
b=hf2.re_hf(0.5)[1]
# np.savetxt('h_real.csv',a, delimiter = ',')
# np.savetxt('h_imag.csv',b, delimiter = ',')
plt.plot()