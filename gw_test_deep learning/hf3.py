# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 15:04:39 2021

@author: 11870
"""
import scipy.integrate as si
import numpy as np
from scipy import *
import numpy as np
import matplotlib.pyplot as plt
import math
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
# def hf(f,zc,z,m1,m2,e0,f0,tc,phic,lambdag,omegabd,S,beta,sigma):
#     M=m1+m2
#     eta=m1*m2/(M**2)
#     Mz=M*pow(eta, 3/5)
#     xz=(np.pi*Mz*f)**(2/3)
#     Ie=pow(np.pi*M*f0,19/9)*e0**2
#     omega=1/omegabd
#     betag=(D(z)*Mz*np.pi**2)/((1+z)*lambdag**2)
#     A=(pow(3,0.5)/2)*pow(f,-7/6)*(pow(Mz,5/6)/(pow(30,0.5)*pow(np.pi,2/3)*DL(z)))

#     expr1=1
#     expr2=(-5/84)*pow(S,2)*omega*pow(xz,-1)
#     expr3=(-128/3)*betag*pow(eta,2/5)*xz
#     expr4=(-2355/1426)*Ie*pow(xz,-19/6)
#     expr5=(3715/756+(55*eta/9))*xz
#     expr6=-4*(4*np.pi-beta)*pow(xz,3/2)
#     expr7=pow(xz,2)*(15293365/508032+27145*eta/504+3085*pow(eta,2)/72-10*sigma)
#     PSI=2*np.pi*f*tc-phic-(np.pi/4)+(3/128)*pow(np.pi*Mz*f,-5/3)*(expr1+expr2+expr3+expr4+expr5+expr6+expr7)
#     h_real=A*np.cos(PSI)
#     h_imag=A*np.sin(PSI)
#     return [h_real,h_imag]

# def f1(x):
#     e1=1/(H0*np.power((0.692+0.308*np.power(1+x,3)),1/2))
#     return e1
# def f2(x):
#     return si.quad(f1,0,x)[0]
# def f3(x):
#     e1=1/(np.power((0.692+0.308*np.power(1+x,3)),1/2)*((1+x)**2))
#     return e1
# def f4(x):
#     return (1+x)*si.quad(f3,0,x)[0]
def hf(f,zc,z,m1,m2,e0,f0,lambdag,omega,beta,sigma,phic,S,tc):
    M=m1+m2
    eta=(m1*m2)/(m1+m2)**2
    Mzc=M*(1+zc)*(eta**(3/5))
    Mz=M*(eta**(3/5))
    x=np.power(np.pi*M*f,2/3)
    Ie=np.power(np.pi*M,19/9)*np.power(e0,2)*np.power(f0,19/9)
    betag=(Mz*67.8*np.pi**2/lambdag**2)*f4(z)
    A=67.8*np.power(Mz,5/6)/(np.power(np.pi,2/3)*30**(1/2)*(1+z)*f2(z))
    psi1=(-3*25/(128*768))*np.power(np.pi*Mzc*f,-5/3)*Mzc*(67.8/2)*(np.pi*M*f)**(-8/3)*(1-
      ((np.power((0.692+0.308*np.power(1+zc,3)),1/2))/(1+zc)))
    psi2=2*np.pi*f*tc-phic-(np.pi/4)+(3/128)*np.power(np.pi*Mz*f,-5/3)*(1-(5/84)*(S**2)*(1/omega)*(1/x)-
      (128*betag*x*eta**(2/5)/3)-(2355*Ie*x**(-19/6)/1462)+x*((3715/756)-(55*eta/9))-4*(4*np.pi-beta)*x**(3/2)+
      ((15293365/508032)+(27145*eta/504)+(3085*(eta**2)/72)-10*sigma)*x**2)
    h__real=(np.power(3,1/2)/2)*A*np.power(f,-7/6)*np.cos(psi1+psi2)*f
    h__imag=(np.power(3,1/2)/2)*A*np.power(f,-7/6)*np.sin(psi1+psi2)*f
    h1__real=(np.power(3,1/2)/2)*A*np.power(f,-7/6)*np.cos(psi2)
    h1__imag=(np.power(3,1/2)/2)*A*np.power(f,-7/6)*np.sin(psi2)
    return [h__real,h__imag,h1__real,h1__imag]












def re_hf(z):
    f=np.arange(10**-5,10**2,0.01)
    n=len(f)
    h=np.empty([n,1],dtype=complex)
    sign_1=0
    sign_1=0

    zc=z

    m1=1.4*10*30
    m2=1.4*10*30
    e0=0
    f0=0.0
    tc=0
    phic=random.uniform(0,2*np.pi)
    lambdag=0.1
    omega=6944.0
    S=0.0
    beta=random.uniform(0,9.4)
    sigma=random.uniform(0,2.5)
    h_real=hf(f,zc,z,m1,m2,e0,f0,lambdag,omega,beta,sigma,phic,S,tc)[0]
    h_imag=hf(f,zc,z,m1,m2,e0,f0,lambdag,omega,beta,sigma,phic,S,tc)[1]


    while sign_1<n:
        h[sign_1]=complex(h_real[sign_1],h_imag[sign_1])
        sign_1+=1
    return [h_real,h_imag]





a=re_hf(1)[0]

b=np.fft.ifft(a).real

plt.subplot(211)
plt.plot(a)
plt.subplot(212)
plt.plot(b)
