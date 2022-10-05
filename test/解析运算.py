# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 17:13:03 2021

@author: 11870
"""

import sympy
from sympy import *
import scipy as si
import latexify
import numpy as np
from scipy import *
import matplotlib.pyplot as plt
import math
import random
import collections

# x,h0,z,y,q,z1 = symbols('x,h0,z,y,q,z1')
# expr1=1/(10000*(h0**2)*(0.692+0.308*(1+z1)**3))
# expr3=10000*(h0**2)*(0.692+0.308*(1+z)**3)
# c1=integrate(expr1,(z1,0,z))
# expr4=1/x
# expr2=(4*pi*((integrate(expr1,(z1,0,z)))**2)*(1+2*z))/(expr3)
# # expr1=expr1.subs(h0,0.678)
# # # plot(c1)
# c2=integrate(expr4,(x,0.1,2))
# prime(c2)

x,h0,z,y,q,z1,a,b,c = symbols('x,h0,z,y,q,z1,a,b,c')
# expr1=1/((10000*(h0**2)*(0.692+0.308*(1+x)**3)))
# expr1=expr1.subs(h0,0.678)
# expr2=integrate(expr1,(x,0,z))
# pprint(expr2)

# pprint(integrate(expr1,(x,0,2)))
# expr1=1/(a+b*(1+x)**3)
# pprint(integrate(expr1,(x,0,z)))
# expr2=integrate(expr1,(x,0,z))
# expr2=simplify(expr2)
# expr3=simplify(expr2)

# def f(x):
#     return 1/(1+2*(1+x)**3)
# v,err=integrate.quad(f,0,2)

# expr1=1/(1+2*(1+x)**3)
# a=integrate(expr1,(x,0,2))
# a=float(a)
# print(a)
# plot(expr1)

# from scipy import integrate
# def f(x):
#     return 1/(1+2*(1+x)**3)
# v, err = integrate.quad(f, 0, 2)# err为误差
# print(v)

# expr1=1/(1+2*(1+x)**3)
# expr2=(integrate(expr1,(x,0,z)))**2
# expr3=(3*expr2*(1+2*z))/(1+z+2*(1+z)**4)

# a=float(integrate(expr3,(z,1,2)))
# print(a)
# # print(a)
# expr3=simplify(expr3)
# # pprint(expr3)
# # print(expr3)
# plot(expr3,(1,2))

# a=root(expr1,2)

# e1=1/(1+x**2)**2
# pprint(integrate(e1,x))


# from sympy import  integrate ,cos,sin
# from sympy.abc import  a,x,y
# import scipy.integrate as si
# import numpy as np
# from scipy import *
# x,n= symbols('x,n')
# expr1=pow(x,4)*pow(sympy.E,x)
# expr2=pow(-1+pow(sympy.E,x),2)
# expr3=expr1/expr2
# expr4=integrate(expr3,(x,0,n))
# pprint(expr4)
# # pprint(simplify(expr4,ratio=100))
# expr5=(x**3)/(-1+pow(sympy.E,x))
# pprint(integrate(expr5,(x,0,n)))


# print(integrate(sin(x)/x,(x,-float("inf"),float("inf"))))

# def f(x):
#     return 4*(x**3)/(-1+np.exp(x))
# print(si.quad(f,0,100000)[0])
# t=np.arange(0,100000,1)
# plt.plot(t,f(t))


x,a,b=symbols('x,a,b')
expr1=a*pow(x,16)+b*pow(x,4)+b*b*pow(x**2,4)
pprint(solve(expr1,x))