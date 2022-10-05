# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 17:15:17 2022

@author: W10
"""

import scipy.integrate as si
import numpy as np
import matplotlib.pyplot as plt
import math


def f1(z):
    e1=67.8*np.power((0.692+0.308*np.power(1+z,3)),1/2)
    e2=1/e1
    return e2

def f2(x):
    e3=np.power(si.quad(f1,0,x)[0],2)
    e4=67.8*np.power((0.692+0.308*np.power(1+x,3)),1/2)*(1+x)
    e5=4*np.pi*e3*(1+2*x)/e4
    return e5

def f3(x):
    e3=np.power(si.quad(f1,0,x)[0],2)
    e4=67.8*np.power((0.692+0.308*np.power(1+x,3)),1/2)*(1+x)
    e6=4*np.pi*e3*(3/4)*(5-x)/e4
    return e6
q=si.quad(f2,0,1)[0]
w=si.quad(f3,1,2)[0]
e=1/(q+w)

def p(x):
    if 0<=x<=1:
        return e*f2(x)
    if 1<x<=2:
        return e*f3(x)
    
    
def re_z(num_1,num_2):
    dn=2/num_2
    mark1=0
    mark2=dn
    mark3=0
    kong1=[]
    kong2=[]
    while mark2<=2:
        mark3=si.quad(p,mark1,mark2)[0]*num_1
        mark4=np.arange(mark1,mark2,dn/mark3)
        for i in mark4:
            kong1.append(i)
        kong2.append(si.quad(p,mark1,mark2)[0])
        mark1+=dn
        mark2+=dn
    kong1=np.array(kong1)
    return [kong1,kong2]
# a=re_z(10000,26)[1]
# b=re_z(10000,26)[0]

