# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 21:42:31 2021

@author: 11870
"""
import numpy as np

import matplotlib.pyplot as plt

def plotdata(x,y,a):
    plt.xlabel('time', fontsize=16) 
    plt.ylabel("signal: "+a, fontsize=16) 
    plt.tick_params(axis='both', which='major', labelsize=16) 
    plt.scatter(x, y,c=y, cmap=plt.cm.Blues,
                edgecolor='none',s=8)

def text_save(filename, data):
    file = open(filename,'a')
    for i in range(len(data)):
        s = str(data[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
        s = s.replace("'",'').replace(',','') +'\n'   #去除单引号，逗号，每行末尾追加换行符
        file.write(s)
    file.close()
    print("保存文件成功")

# def text_save_two(filename,data1,data2):
#     file = open(filename,'a')
#     for i in range(len(data1)):
#         s = str(data1[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
#         s = s.replace("'",'').replace(',','') +'\n'   #去除单引号，逗号，每行末尾追加换行符
#         file.write(s)
#     for i in range(len(data2)):
#         s = str(data2[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
#         s = s.replace("'",'').replace(',','') +'\n'   #去除单引号，逗号，每行末尾追加换行符
#         file.write(s)
#     file.close()
#     print("保存文件成功")

def npy_save(filename,data):
    numpy_array = np.array(data)
    np.save(filename+'.npy',numpy_array )
    
def merge(ht,hn):
    c=ht[:]
    for a in hn:
        c.append(a)
    return c

t = [float(row.split()[0]) for row in open("0_data.txt")]
H_zhn_zht = [float(row.split()[1]) for row in open("0_data.txt")]
# plotdata(t,H_zhn_zht,'H_zhn_zht')


L_zhn= [float(row.split()[3]) for row in open("0_data.txt")]
L_zht= [float(row.split()[4]) for row in open("0_data.txt")]
L_zhn_zht=[a + b for a, b in zip(L_zhn, L_zht)]
# plotdata(t,L_zhn,'L_zhn')
# plotdata(t,L_zht,'L_zht')
# plotdata(t,L_zhn_zht,'L_zhn_zht')


V_zhn= [float(row.split()[5]) for row in open("0_data.txt")]
V_zht= [float(row.split()[6]) for row in open("0_data.txt")]
V_zhn_zht=[a + b for a, b in zip(V_zhn, V_zht)]
# plotdata(t,V_zhn,'V_zhn')
# plotdata(t,V_zht,'V_zht')
# plotdata(t,V_zhn_zht,'V_zhn_zht')    

L_hnht=merge(L_zht,L_zhn)
V_hnht=merge(V_zht,V_zhn)

# text_save('L_hnht.txt', L_hnht)
# text_save('V_hnht.txt', V_hnht)

# npy_save('L_hnht', L_hnht)
# npy_save('V_hnht.txt', V_hnht)








      




