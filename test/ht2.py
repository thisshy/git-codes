# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 18:06:19 2022

@author: W10
"""

import scipy.integrate as si
import numpy as np
from scipy import *
import matplotlib.pyplot as plt



# //////图片的信息标签可以输入中文的代码如下
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams['axes.unicode_minus'] = False

H0=(67.8*10**-19)/3.086
G=6.67*10**-11
c=3.0*10**8

def H(z):
    e1=H0*(0.692+0.308*pow(1+z, 3))**0.5
    return e1
def f1(z):
    return 1/H(z)
def f2(z):
    return 1/(H(z)*((1+z)**2))
def f3(z):
    return si.quad(f1,0,z)[0]
def f4(z):
    return si.quad(f2,0,z)[0]
def D(z):
    return (1+z)*f4(z)*c
def DL(z):
    return (1+z)*f3(z)*c
def hf(f,zc,z,m1,m2,tc,S,phic,lambdag,omegabd):
    # beta=np.random.uniform(0,9.4)
    # sigma=np.random.uniform(0,2.5)
    beta=0
    sigma=0
    M=m1+m2
    eta=m1*m2/(M**2)
    Mzc=M*(1+zc)*pow(eta, 3/5)
    Mz=M*pow(eta, 3/5)
    xzc=(np.pi*Mzc*f)**(2/3)
    x=pow(np.pi*Mz*f,-5/3)*pow(G,-5/3)*pow(c,5)
    x1=pow(np.pi*Mz*f,1)
    omega=1/omegabd
    betag=D(z)*Mz*pow(np.pi,2)*pow((1+z)*(lambdag**2),-1)*pow(G,1)*pow(c,-2)
    PSIN=(3/128)*pow(np.pi*Mzc*f,-5/3)
    X=(H0/2)*(1-(H(zc)/((1+zc)*H0)))
    PSIacc=-PSIN*X*(25/768)*Mzc*pow(xzc,-4)*pow(G, -10/3)*pow(c, 10)
    
    
    A=pow(pow(30,1/2)*pow(np.pi,2/3),-1)*pow(Mz,5/6)*pow(DL(z),-1)*pow(G,5/6)*pow(c,-3/2)*pow(f,-7/6)*pow(3,1/2)/2
    expr1=1
    expr2=(5/84)*pow(S,2)*pow(omegabd,-1)*pow(eta,2/5)*pow(x1,-2/3)*pow(G,-2/3)*pow(c,2)
    expr3=(128/3)*betag*pow(x1,2/3)*pow(G,2/3)*pow(c,-2)
    expr4=(3715/756+55/9*eta)*pow(eta,-2/5)*pow(x1,2/3)*pow(G,2/3)*pow(c,-2)
    expr5=16*np.pi*pow(eta,-3/5)*pow(x1,1)*pow(G,1)*pow(c,-3)
    expr6=4*beta*pow(eta,-3/5)*pow(x1,1)*pow(G,1)*pow(c,-3)
    expr7=(15293365/508032+27145*eta/504+pow(eta,2)*3085/72)*pow(eta,-4/5)*pow(x1,4/3)*pow(G,4/3)*pow(c,-4)
    expr8=10*sigma*pow(eta,-4/5)*pow(x1,4/3)*pow(G,4/3)*pow(c,-4)
    PSI=2*np.pi*f*tc-phic+(3/128)*x*(expr1-expr2-expr3+expr4-expr5+expr6+expr7-expr8)
    h_real=A*np.cos(PSIacc+PSI)
    h_imag=A*np.sin(PSIacc+PSI)
    h1_real=A*np.cos(PSI)
    h1_imag=A*np.sin(PSI)
    
    return [h_real,h_imag,h1_real,h1_imag,X,A]


def re_hf(f,z):
    
    sign_1=0
    sign_2=0
    zc=z
    m1=1.4*1.989*(10**35)
    m2=1.4*1.989*(10**35)
    tc=0.0
    S=0.0
    phic=0.0
    lambdag=1.1*10**21
    omegabd=96944.0
    h_real=hf(f,zc,z,m1,m2,tc,S,phic,lambdag,omegabd)[0]
    h_imag=hf(f,zc,z,m1,m2,tc,S,phic,lambdag,omegabd)[1]
    h1_real=hf(f,zc,z,m1,m2,tc,S,phic,lambdag,omegabd)[2]
    h1_imag=hf(f,zc,z,m1,m2,tc,S,phic,lambdag,omegabd)[3]
    n=len(f)
    h=[]
    h1=[]
    while sign_1<n:
        h.append(complex(h_real[sign_1],h_imag[sign_1]))
        sign_1+=1
    while sign_2<n:
        h1.append(complex(h1_real[sign_2],h1_imag[sign_2]))
        sign_2+=1
    # h=np.array(h)
    # h1=np.array(h1)
    return[h,h1,h1_real]


def re_A(f,z):
    
    sign_1=0
    sign_2=0
    zc=z
    m1=1.4*1.989*(10**40)
    m2=1.4*1.989*(10**40)
    tc=0.0
    S=0.0
    phic=0.0
    lambdag=1.1*10**21
    omegabd=96944.0
    A=hf(f,zc,z,m1,m2,tc,S,phic,lambdag,omegabd)[5]
    return np.array(A)


##########################################################判断如何求最敏感频率
# M=1.4*1.989*(10**35)
# f_up=pow(pow(3,3/2)*np.pi*M*2,-1)*pow(c,3)*pow(G,-1)








def re_X(z):
    X=(H0/2)*(1-(H(z)/((1+z)*H0)))
    return X



def re_ht(f,z):
    # f=np.arange(10**-3,1,0.000001)
    kong1=re_hf(f,z)[0]
    kong2=re_hf(f,z)[1]
    kong4=[]
    kong4=kong1[::-1]
    kong1=np.array(kong1)
    kong4=np.array(kong4)
    h=np.hstack((kong1,kong4))
    ifft_h=np.fft.ifft(h).real
    n=len(f)
    n1=int(0.5*n)
    n2=-50
    ifft_h=ifft_h.tolist()
    ifft_h=ifft_h[n1:n2]
    ifft_h=np.array(ifft_h)
    return ifft_h




# f=np.arange(10**-3,1,10**-6)
# a=np.array(re_hf(f,1)[0])

# b=re_ht(f,1)







