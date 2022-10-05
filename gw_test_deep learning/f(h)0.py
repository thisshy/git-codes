# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 10:53:36 2021

@author: 11870
"""

import scipy.integrate as si
from scipy import *
import numpy as np	
import matplotlib.pyplot as plt
import random
import collections






# //////图片的信息标签可以输入中文的代码如下
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams['axes.unicode_minus'] = False  

H0=(67.8*10**-19)/3.086
G=6.67*10**-11
c=3.0*10**8

def H(x):
    e1=H0*(0.692+0.308*pow(1+x, 3))**0.5
    return e1
def f1(x):
    return 1/H(x)
def f2(x):
    return 1/(H(x)*(1+x)**2)
def f3(x):
    return si.quad(f1,0,x)[0]
def f4(x):
    return si.quad(f2,0,x)[0]
def D(x):
    return (1+x)*f4(x)
def DL(x):
    return (1+x)*f3(x)
def hf(f,zc,z,m1,m2,e0,f0,tc,phic,lambdag,omegabd,S,beta,sigma):
    M=m1+m2
    eta=m1*m2/(M**2)
    Mzc=M*(1+zc)*pow(eta, 3/5)
    Mz=M*pow(eta, 3/5)
    xzc=(np.pi*Mzc*f)**(2/3)*pow(c,-2)*pow(G,2/3)
    xz=(np.pi*Mz*f)**(2/3)*pow(c,-2)*pow(G,2/3)
    Ie=pow(np.pi*M*f0,19/9)*e0**2*pow(G,19/9)*pow(c,-19/3)
    omega=1/omegabd
    betag=(D(z)*Mz*np.pi**2)/((1+z)*lambdag**2)*pow(G,1)*pow(c,-2)
    PSIN=(3/128)*pow(np.pi*Mzc*f,-5/3)
    X=(H0/2)*(1-(H(zc)/((1+zc)*H0)))
    A=(pow(3,0.5)/2)*pow(f,-7/6)*(pow(Mz,5/6)/(pow(30,0.5)*pow(np.pi,2/3)*DL(z)))*pow(G,5/6)*pow(c,-3/2)
    PSIacc=-PSIN*X*(25/768)*Mzc*pow(xzc,-4)*pow(G, -10/3)*pow(c, 10)
    expr1=1
    expr2=(-5/84)*pow(S,2)*omega*pow(xz,-1)
    expr3=(-128/3)*betag*pow(eta,2/5)*xz
    expr4=(-2355/1426)*Ie*pow(xz,-19/6)
    expr5=(3715/756+(55*eta/9))*xz
    expr6=-4*(4*np.pi-beta)*pow(xz,3/2)
    expr7=pow(xz,2)*(15293365/508032+27145*eta/504+3085*pow(eta,2)/72-10*sigma)
    PSI=2*np.pi*f*tc-phic-(np.pi/4)+(3/128)*pow(np.pi*Mz*f,-5/3)*(expr1+expr2+expr3+expr4+expr5+expr6+expr7)*pow(G,-5/3)*pow(c,5)
    h_real=A*np.cos(PSIacc+PSI)
    h_imag=A*np.sin(PSIacc+PSI)
    h1_real=A*np.cos(PSI)
    h1_imag=A*np.sin(PSI)
    # print(h_real)
    return [h_real,h_imag,h1_real,h1_imag]

f=np.arange(0.0001,1,0.01)
zc=1.0
z=1.0
m1=1.989*10**30
m2=1.989*10**30
e0=0.5
f0=0.003
tc=1.0
phic=0.0
lambdag=3.1*10**19
omegabd=6944.0
S=0.0
beta=2.0
sigma=2.0
hhh=hf(f,zc,z,m1,m2,e0,f0,tc,phic,lambdag,omegabd,S,beta,sigma)[0]
b=[np.cos(f),1]

def ff(f,c,d):
    return[1/(np.power(np.cos(f*c+d),2)),c*f,d*f**2]
a=ff(f,1,1)
dd=np.cos(f)

print(hhh[np.arange(int(30))])