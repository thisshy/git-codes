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

def f4(x):
    return si.quad(p,0,x)[0]


# ////////////////////////////////////////////////////////////////////////
# 画分布函数图像
# plt.subplot(223)
# r1=[]
# t1=[]
# s1=0
# d1=0
# while s1<=500:
#     r1.append(random.uniform(0,2))   
#     s1+=1
    
# r1.sort()
# while d1<=500:
#     t1.append(f4(r1[d1]))
#     d1+=1
    
# plt.scatter(t1, r1, s=0.5)
# plt.xlabel('probability', fontsize=25)
# plt.ylabel('z', fontsize=25)
# plt.title('Distribution function-z',fontsize=25)


#///////////////////////////////////////////////////////////////////////////
# 画z p(z)函数图像

plt.subplot(232)
a=b=100
q=[]
w=[]
a1=0
a2=0
while a>=0:
    a2=a1+2/b
    if a2>2:
        break
    q.append((a1+a2)/2)
    w.append(f4(a2)-f4(a1))
    a1+=2/b
    a-=1
print(len(q),len(w))
plt.scatter(q,w,s=5)
plt.xlabel('z', fontsize=25)
plt.ylabel('p(z)', fontsize=25)
plt.title('z-p(z)',fontsize=25)
plt.show()
# print(sum(w))
# 0+2/b是一个积分区间元，中间值是(a1+a2)/2,比如0-0.2，中间值是0.1，第二个中间值
# 是第一个中间值加一个区间元，中间值往两边加减 （区间元)/2就是落在这个中间值附近区间
# 范围，比如，b=10，则区间元为2/10=0.2，所以第一个中间值就是0.1，对应的区间实际上是
# （0.1-0.1，0.1+0.1）,同理第二个中间值是0.3，对应的区间是（0.3-0.1,0.3+0.1)

#////////////////////////////////////////////////////////////////////////////

#   # 画 d（z）函数的图像
# def f5(z):
#     return 1/(67.8*(0.692+0.308*(1+z)**3))**0.5
# def f6(x):
#     return (1+x)*si.quad(f5,0,x)[0]

# r=[]
# t=[]
# s=0
# d=0
# while s<=10:
#     r.append(random.uniform(0,1))   
#     s+=1
    
# r.sort()
# while d<=10:
#     t.append(f6(r[d]))
#     d+=1
    
# plt.scatter(r, t, s=0.5)	
# plt.show()

# /////////////////////////////////////////////////////////////////////////















