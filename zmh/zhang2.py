# -*- coding: utf-8 -*-
"""
Created on Sat May 21 15:24:13 2022

@author: Lenovo
"""
import scipy.integrate as si
import numpy as np
from scipy import *
import matplotlib.pyplot as plt

def S(v,r):
    A=0.7
    M=0.3
    vE=pow(3/7,1/3)*2*pow(10,-18)
    vH=2*pow(10,-18)
    v2=51*pow(3/7,1/3)*2*pow(10,-18)
    vS=pow(10,8)
    e0=8.854187817*pow(10,-12)
    B=pow(A/M,1/3)
    p=-1.9
    ps=-0.552
    e=(1+p)*(1-r)/r
    a=0.37*pow(10,-5)*(A/M)
    y2=B*pow(r,-1)*pow(3454,-0.5)
    Ex=25
    Ey=25*pow(3,0.5)
    Ez=50*pow(3,0.5)
    Bx=10
    By=10
    Bz=10*pow(2,0.5)
    c=3*pow(10,8)
    z=0.1
    S1=0
    if 0.625*5.5*c<=v<0.625*6.5*c:
        h=a*pow(vS/vH,ps)*(vH/v2)*pow(v/vH,p-ps+1)*pow(B,-3-e)*np.cos(2*np.pi*v*y2)
        ex1=(1/2)*2*np.pi*(v/c)*z*h*((1/2)*Ex+Ey+Ez)
        ex2=(1/2)*2*np.pi*v*z*h*(Bx-(3/2)*By+Bz)
        EX=ex1+ex2
        S1=(1/2)*e0*c*pow(EX,2)*pow(10,10)*np.exp(-abs((v-0.625*6*c)*pow(10,-7)))
    if 0.625*6.5*c<=v<0.625*7.5*c:
        h=a*pow(vS/vH,ps)*(vH/v2)*pow(v/vH,p-ps+1)*pow(B,-3-e)*np.cos(2*np.pi*v*y2)
        ex1=(1/2)*2*np.pi*(v/c)*z*h*((1/2)*Ex+Ey+Ez)
        ex2=(1/2)*2*np.pi*v*z*h*(Bx-(3/2)*By+Bz)
        EX=ex1+ex2
        S1=(1/2)*e0*c*pow(EX,2)*pow(10,10)*np.exp(-abs((v-0.625*7*c)*pow(10,-7)))
    if 0.625*7.5*c<=v<0.625*8.5*c:
        h=a*pow(vS/vH,ps)*(vH/v2)*pow(v/vH,p-ps+1)*pow(B,-3-e)*np.cos(2*np.pi*v*y2)
        ex1=(1/2)*2*np.pi*(v/c)*z*h*((1/2)*Ex+Ey+Ez)
        ex2=(1/2)*2*np.pi*v*z*h*(Bx-(3/2)*By+Bz)
        EX=ex1+ex2
        S1=(1/2)*e0*c*pow(EX,2)*pow(10,10)*np.exp(-abs((v-0.625*8*c)*pow(10,-7)))
    if 0.625*8.5*c<=v<0.625*9.5*c:
        h=a*pow(vS/vH,ps)*(vH/v2)*pow(v/vH,p-ps+1)*pow(B,-3-e)*np.cos(2*np.pi*v*y2)
        ex1=(1/2)*2*np.pi*(v/c)*z*h*((1/2)*Ex+Ey+Ez)
        ex2=(1/2)*2*np.pi*v*z*h*(Bx-(3/2)*By+Bz)
        EX=ex1+ex2
        S1=(1/2)*e0*c*pow(EX,2)*pow(10,10)*np.exp(-abs((v-0.625*9*c)*pow(10,-7)))
    if 0.625*9.5*c<=v<0.625*10.5*c:
        h=a*pow(vS/vH,ps)*(vH/v2)*pow(v/vH,p-ps+1)*pow(B,-3-e)*np.cos(2*np.pi*v*y2)
        ex1=(1/2)*2*np.pi*(v/c)*z*h*((1/2)*Ex+Ey+Ez)
        ex2=(1/2)*2*np.pi*v*z*h*(Bx-(3/2)*By+Bz)
        EX=ex1+ex2
        S1=(1/2)*e0*c*pow(EX,2)*pow(10,10)*np.exp(-abs((v-0.625*10*c)*pow(10,-7)))
    if 0.625*10.5*c<=v<0.625*11.5*c:
        h=a*pow(vS/vH,ps)*(vH/v2)*pow(v/vH,p-ps+1)*pow(B,-3-e)*np.cos(2*np.pi*v*y2)
        ex1=(1/2)*2*np.pi*(v/c)*z*h*((1/2)*Ex+Ey+Ez)
        ex2=(1/2)*2*np.pi*v*z*h*(Bx-(3/2)*By+Bz)
        EX=ex1+ex2
        S1=(1/2)*e0*c*pow(EX,2)*pow(10,10)*np.exp(-abs((v-0.625*11*c)*pow(10,-7)))    
    if 0.625*11.5*c<=v<0.625*12.5*c:
        h=a*pow(vS/vH,ps)*(vH/v2)*pow(v/vH,p-ps+1)*pow(B,-3-e)*np.cos(2*np.pi*v*y2)
        ex1=(1/2)*2*np.pi*(v/c)*z*h*((1/2)*Ex+Ey+Ez)
        ex2=(1/2)*2*np.pi*v*z*h*(Bx-(3/2)*By+Bz)
        EX=ex1+ex2
        S1=(1/2)*e0*c*pow(EX,2)*pow(10,10)*np.exp(-abs((v-0.625*12*c)*pow(10,-7)))
    if 0.625*12.5*c<=v<0.625*13.5*c:
        h=a*pow(vS/vH,ps)*(vH/v2)*pow(v/vH,p-ps+1)*pow(B,-3-e)*np.cos(2*np.pi*v*y2)
        ex1=(1/2)*2*np.pi*(v/c)*z*h*((1/2)*Ex+Ey+Ez)
        ex2=(1/2)*2*np.pi*v*z*h*(Bx-(3/2)*By+Bz)
        EX=ex1+ex2
        S1=(1/2)*e0*c*pow(EX,2)*pow(10,10)*np.exp(-abs((v-0.625*13*c)*pow(10,-7)))    
    if 0.625*13.5*c<=v<0.625*14.5*c:
        h=a*pow(vS/vH,ps)*(vH/v2)*pow(v/vH,p-ps+1)*pow(B,-3-e)*np.cos(2*np.pi*v*y2)
        ex1=(1/2)*2*np.pi*(v/c)*z*h*((1/2)*Ex+Ey+Ez)
        ex2=(1/2)*2*np.pi*v*z*h*(Bx-(3/2)*By+Bz)
        EX=ex1+ex2
        S1=(1/2)*e0*c*pow(EX,2)*pow(10,10)*np.exp(-abs((v-0.625*14*c)*pow(10,-7)))
    if 0.625*14.5*c<=v<0.625*15.5*c:
        h=a*pow(vS/vH,ps)*(vH/v2)*pow(v/vH,p-ps+1)*pow(B,-3-e)*np.cos(2*np.pi*v*y2)
        ex1=(1/2)*2*np.pi*(v/c)*z*h*((1/2)*Ex+Ey+Ez)
        ex2=(1/2)*2*np.pi*v*z*h*(Bx-(3/2)*By+Bz)
        EX=ex1+ex2
        S1=(1/2)*e0*c*pow(EX,2)*pow(10,10)*np.exp(-abs((v-0.625*15*c)*pow(10,-7)))   
    if 0.625*15.5*c<=v<0.625*16.5*c:
        h=a*pow(vS/vH,ps)*(vH/v2)*pow(v/vH,p-ps+1)*pow(B,-3-e)*np.cos(2*np.pi*v*y2)
        ex1=(1/2)*2*np.pi*(v/c)*z*h*((1/2)*Ex+Ey+Ez)
        ex2=(1/2)*2*np.pi*v*z*h*(Bx-(3/2)*By+Bz)
        EX=ex1+ex2
        S1=(1/2)*e0*c*pow(EX,2)*pow(10,10)*np.exp(-abs((v-0.625*16*c)*pow(10,-7)))   
    if 0.625*16.5*c<=v<0.625*17.5*c:
        h=a*pow(vS/vH,ps)*(vH/v2)*pow(v/vH,p-ps+1)*pow(B,-3-e)*np.cos(2*np.pi*v*y2)
        ex1=(1/2)*2*np.pi*(v/c)*z*h*((1/2)*Ex+Ey+Ez)
        ex2=(1/2)*2*np.pi*v*z*h*(Bx-(3/2)*By+Bz)
        EX=ex1+ex2
        S1=(1/2)*e0*c*pow(EX,2)*pow(10,10)*np.exp(-abs((v-0.625*17*c)*pow(10,-7)))    
    if 0.625*17.5*c<=v<0.625*18.5*c:
        h=a*pow(vS/vH,ps)*(vH/v2)*pow(v/vH,p-ps+1)*pow(B,-3-e)*np.cos(2*np.pi*v*y2)
        ex1=(1/2)*2*np.pi*(v/c)*z*h*((1/2)*Ex+Ey+Ez)
        ex2=(1/2)*2*np.pi*v*z*h*(Bx-(3/2)*By+Bz)
        EX=ex1+ex2
        S1=(1/2)*e0*c*pow(EX,2)*pow(10,10)*np.exp(-abs((v-0.625*18*c)*pow(10,-7)))
    if 0.625*18.5*c<=v<0.625*19.5*c:
        h=a*pow(vS/vH,ps)*(vH/v2)*pow(v/vH,p-ps+1)*pow(B,-3-e)*np.cos(2*np.pi*v*y2)
        ex1=(1/2)*2*np.pi*(v/c)*z*h*((1/2)*Ex+Ey+Ez)
        ex2=(1/2)*2*np.pi*v*z*h*(Bx-(3/2)*By+Bz)
        EX=ex1+ex2
        S1=(1/2)*e0*c*pow(EX,2)*pow(10,10)*np.exp(-abs((v-0.625*19*c)*pow(10,-7)))   
    if 0.625*19.5*c<=v<0.625*20.5*c:
        h=a*pow(vS/vH,ps)*(vH/v2)*pow(v/vH,p-ps+1)*pow(B,-3-e)*np.cos(2*np.pi*v*y2)
        ex1=(1/2)*2*np.pi*(v/c)*z*h*((1/2)*Ex+Ey+Ez)
        ex2=(1/2)*2*np.pi*v*z*h*(Bx-(3/2)*By+Bz)
        EX=ex1+ex2
        S1=(1/2)*e0*c*pow(EX,2)*pow(10,10)*np.exp(-abs((v-0.625*20*c)*pow(10,-7)))
    if 0.625*20.5*c<=v<=0.625*21.5*c:
        h=a*pow(vS/vH,ps)*(vH/v2)*pow(v/vH,p-ps+1)*pow(B,-3-e)*np.cos(2*np.pi*v*y2)
        ex1=(1/2)*2*np.pi*(v/c)*z*h*((1/2)*Ex+Ey+Ez)
        ex2=(1/2)*2*np.pi*v*z*h*(Bx-(3/2)*By+Bz)
        EX=ex1+ex2
        S1=(1/2)*e0*c*pow(EX,2)*pow(10,10)*np.exp(-abs((v-0.625*21*c)*pow(10,-7)))    
    return S1





def re_s(v,r):
    n1=0
    n2=len(v)
    kong=[]
    while n1<n2:
        kong.append(S(v[n1],r))
        n1+=1
    kong1=np.array(kong)
    return kong1




r=np.linspace(1.04,1.07,500)


def re_data(r):
    v=np.linspace(0.625*5.5*3*pow(10,8),0.625*21.5*3*pow(10,8),1000)
    kong=[]
    n1=0
    n2=len(r)
    while n1<n2:
        kong.append(re_s(v,r[n1]))
        n1+=1
    return kong

data=np.log10(re_data(r))
