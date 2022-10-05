# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 15:24:56 2021

@author: 11870
"""
import produce_z
import hf2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



seed=1
a=produce_z.return_z(8192)

b=np.empty(len(a),dtype=float)
for i in np.arange(len(a)):
    b[i]=hf2.DL(a[i])
c=np.ones(len(a),dtype=float)
d=hf2.X(a)
max1=np.max(b)
max2=np.max(d)
dl=b/max1
x=d/max2
x.reshape(len(a),1)

#字典中的key值即为csv中列名
dataframe = pd.DataFrame({'a':a,'b':dl,'c':c,'d':x})
 
#将DataFrame存储为csv,index表示是否显示行名，default=True
dataframe.to_csv("test.csv",index=False,sep=',')
def re_maxv():
    return max2


