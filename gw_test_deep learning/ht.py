# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 18:06:19 2022

@author: W10
"""
import matplotlib
import scipy.integrate as si
import numpy as np
from scipy import *
import matplotlib.pyplot as plt
import sf


# //////图片的信息标签可以输入中文的代码如下
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams['axes.unicode_minus'] = False

H0=(70*10**-19)/3.086
G=6.67*10**-11
c=2.99792458*pow(10,8)

def H(z):
    e1=H0*pow(0.692+0.308*pow(1+z, 3),0.5)
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
    beta=2
    sigma=2
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
    expr4=(3715/756+55*eta/9)*pow(eta,-2/5)*pow(x1,2/3)*pow(G,2/3)*pow(c,-2)
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
    m1=1.4*1.989*(10**30)*pow(10,0)
    m2=1.4*1.989*(10**30)*pow(10,0)
    tc=0
    S=0.0
    phic=0
    lambdag=1.1*10**21
    omegabd=9.0*pow(10,4)
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


def re_ht(f,z,N,tmax):
    h=np.array(re_hf(f,z)[0])*pow(N,1/2)/(2*np.pi)
    ifft_h=np.fft.ifft(h,N).real
    kong=ifft_h[::-1]
    # n=len(f)
    # n1=int(0.5*N)
    # n2=-1
    # ifft_h=ifft_h.tolist()
    # ifft_h=ifft_h[n1:n2]

    return [kong,h]


def re_ht1(f,z,N,tmax):
    h=np.array(re_hf(f,z)[1])*pow(N,1/2)/(2*np.pi)
    ifft_h=np.fft.ifft(h,N).real
    kong=ifft_h[::-1]
    # n=len(f)
    # n1=int(0.5*N)
    # n2=-1
    # ifft_h=ifft_h.tolist()
    # ifft_h=ifft_h[n1:n2]

    return [kong,h]



##########################################################判断如何求最敏感频率
# M=1.4*1.989*(10**35)
# f_up=pow(pow(3,3/2)*np.pi*M*2,-1)*pow(c,3)*pow(G,-1)




# /////////////////////////////////////////////////求merge频率

m1=1.4*1.989*(10**30)*pow(10,0)
m2=1.4*1.989*(10**30)*pow(10,0)
M12=m1+m2
fmerge=pow(np.pi*M12*pow(6,3/2),-1)*pow(G,-1)*pow(c,3)






fmin=0.4
fmax=1
M=20000000
f=np.linspace(fmin,fmax,M)
dt=pow(2*fmax,-1)
tmax=(M-1)*dt
t=np.linspace(0,tmax,M)

z=1
a=re_ht(f,z,M,tmax)[0]
b=re_hf(f,z)[0]
a1=re_ht1(f,z,M,tmax)[0]


# plt.subplot(211)
# plt.plot(t,a)
# plt.subplot(212)
# plt.plot(f,b)


plt.plot(t,a)
plt.plot(t,a1)

# # # //////////////////////////////////////////fin
m1=1.4*1.989*(10**30)*pow(10,0)
m2=1.4*1.989*(10**30)*pow(10,0)
M12=m1+m2
eta=m1*m2/(M12**2)
zc=1
Mzc=M12*(1+zc)*pow(eta, 3/5)
fin=pow(256/5,-3/8)*(1/np.pi)*pow(Mzc,-5/8)*pow(tmax,-3/8)*pow(G,-5/8)*pow(c,15/8)










