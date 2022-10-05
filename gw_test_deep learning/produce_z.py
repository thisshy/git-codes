# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 17:07:21 2021

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



def f1(z):
    e1=67.8*np.power((0.692+0.308*np.power(1+z,3)),1/2)
    e2=1/e1
    return e2

def f2(x):
    e3=np.power(si.quad(f1,0,2*x)[0],2)
    e4=67.8*np.power((0.692+0.308*np.power(1+2*x,3)),1/2)*(1+2*x)
    e5=4*np.pi*e3*(1+4*x)/e4
    return e5

def f3(x):
    e3=np.power(si.quad(f1,0,2*x)[0],2)
    e4=67.8*np.power((0.692+0.308*np.power(1+2*x,3)),1/2)*(1+2*x)
    e6=4*np.pi*e3*(3/4)*(5-2*x)/e4
    return e6
q=si.quad(f2,0,0.5)[0]
w=si.quad(f3,0.5,1)[0]
e=1/(q+w)

def p(x):
    if 0<=x<=0.5:
        return e*f2(x)
    if 0.5<x<=1:
        return e*f3(x)

def f4(x):
    return si.quad(p,0,x)[0]


# ///////////////////////////////////////////////////////////////////////////////////////


# ////////////////////////////////////////////////////////////////////


# ////////////////////////////////////////////////////////////////////////



# ///////////////////////////////////////////////////////////////////////
# b是总的点数，count_dict字典中记录了在20个区间内分别产生了多少个点，将每个区间的点数除总的点数
# 就是在每个区间产生点的概率，因此g1中记录了各个区间产生点的概率。

# plt.subplot(223)
# x_list= sorted([(key + 0.5) / bin_count for key in count_dict.keys()])
# x_nexlist = [i*2 for i in x_list]
# dd = collections.OrderedDict(sorted(count_dict.items()))
# g1=[]
# for k,v in dd.items():
#     g1.append(v/b)
# plt.scatter(x_nexlist, g1, s=10)
# plt.xlabel('z', fontsize=10)
# plt.ylabel('在相应区间产生z的概率',fontsize=10)
# plt.title('z的区间概率计算',fontsize=10)
# # plt.show()
# print(sum(g1))





# ////////////////////////////////////////////////////////////////////////////////////


# ////////////////////////////////////////////////////////////////////////////////

# 求函数的导函数，数值法

# from scipy.misc import derivative
# def f6(x):
#     return derivative(f5,x,dx=1e-6)
# def f7(x):
#     return derivative(f6,x,dx=1e-6)


# r=[]
# t=[]
# s=0
# d=0
# while s<=30:
#     r.append(random.uniform(0,1))   
#     s+=1
    
# r.sort()
# while d<=30:
#     if 2<=f7(r[d]) or f7(r[d])<=0:
#         t.append(0 )
#     else:
#         t.append(f7(r[d]))
#     d+=1
# print(r,t)
# plt.scatter(r, t, s=0.5)	
# plt.show()

# ///////////////////////////////////////////////////////////////////////////////////////



# ////////////////////////////////////////////////////////////////////////
# 画分布函数图像
# plt.subplot(221)
# kong_1=[]
# kong_2=[]
# sign_1=0
# sign_2=0
# while sign_1<=5000:
#     kong_1.append(random.uniform(0,2))   
#     sign_1+=1
    
# kong_1.sort()
# while sign_2<=5000:
#     kong_2.append(fd(kong_1[sign_2]))
#     sign_2+=1
    
# plt.scatter(kong_2,kong_1, s=0.5)
# plt.xlabel('probability', fontsize=10)
# plt.ylabel('z', fontsize=10)
# plt.title('Distribution function-z',fontsize=10)


#///////////////////////////////////////////////////////////////////////////
# 画z P(z)函数图像

# plt.subplot(222)
# a=b=1000
# kong_3=[]
# kong_4=[]
# sign_3=0
# sign_4=0
# while a>=0:
#     sign_4=sign_3+2/b
#     if sign_4>2:
#         break
#     kong_3.append((sign_3+sign_4)/2)
#     kong_4.append(fd(sign_4)-fd(sign_3))
#     sign_3+=2/b
#     a-=1

# plt.scatter(kong_3,kong_4,s=5)
# plt.xlabel('z', fontsize=10)
# plt.ylabel('P(z)', fontsize=10)
# plt.title('z-P(z)',fontsize=10)
# plt.show()
# print(sum(w))
# 0+2/b是一个积分区间元，中间值是(sign_3+sign_4)/2,比如0-0.2，中间值是0.1，第二个中间值
# 是第一个中间值加一个区间元，中间值往两边加减 （区间元)/2就是落在这个中间值附近区间
# 范围，比如，b=10，则区间元为2/10=0.2，所以第一个中间值就是0.1，对应的区间实际上是
# （0.1-0.1，0.1+0.1）,同理第二个中间值是0.3，对应的区间是（0.3-0.1,0.3+0.1)

#////////////////////////////////////////////////////////////////////////////

  # 画 d（z）函数的图像
def f5(z):
    return 1/(67.8*(0.692+0.308*(1+z)**3))**0.5
def f6(x):
    return (1+x)*si.quad(f5,0,x)[0]
def f7(x):
    return si.quad(f6,0,x)[0]
r=[]
t=[]
s=0
d=0
while s<=10:
    r.append(random.uniform(0,1))   
    s+=1
    
r.sort()
while d<=10:
    t.append(f7(r[d]))
    d+=1
    
plt.scatter(r, t, s=0.5)	
plt.show()

# /////////////////////////////////////////////////////////////////////////






