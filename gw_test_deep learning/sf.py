# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 21:03:54 2021

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

# c=np.c_[a,b]

def S_decigo(f):
    expr1=1.3*pow(f,-4)*pow(10,-52)
    expr2=4.6*pow(10,-49)
    expr3=2.0*pow(f,2)*pow(10,-49)
    return expr1+expr2+expr3
def S_gal(f):
    expr1=2.1*pow(10,-45)*pow(f,-7/3)
    return expr1
def S_exgal(f):
    expr1=4.2*pow(10,-47)*pow(f,-7/3)
    return expr1
def S_ns(f):
    expr1=1.3*pow(10,-48)*pow(f,-7/3)
    return expr1
def S_sum(f):
    K=4.5
    dn_df=2*pow(10,-3)*pow(f,-11/3)
    F=np.exp(-2*pow(f/0.05,2))
    T=pow(10,7)*3.1536
    expr0=S_exgal(f)*F+0.01*S_ns(f)
    expr1=(S_decigo(f))/(np.exp(-K*dn_df*(1/T)))+expr0
    expr2=S_decigo(f)+S_gal(f)*F+expr0
    expr3=np.c_[expr1,expr2]
    expr4=np.min(expr3,axis=1)
    return expr4


def re_sf(f):
    return S_sum(f)




