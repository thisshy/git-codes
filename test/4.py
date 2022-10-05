# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 21:42:42 2021

@author: 11870
"""
def Model_Predictive(filename):
    Q=W=E=R=Z=X=C=V=q=w=e=r=z=x=c=v=0
    a=[float(row.split()[0]) for row in open(filename)]
    b=[float(row.split()[1]) for row in open(filename)]
    for i in range(len(a)):
        if a[i]==0:
            q+=1
            if b[i]==1:
               Q+=1
        if 0<a[i]<=15:
            w+=1
            if b[i]==1:
               W+=1
        if 15<a[i]<=20:
            e+=1
            if b[i]==1:
               E+=1
        if 20<a[i]<=25:
            r+=1
            if b[i]==1:
               R+=1
        if 25<a[i]<=30:
            z+=1
            if b[i]==1:
               Z+=1
        if 30<a[i]<=35:
            x+=1
            if b[i]==1:
               X+=1
        if 35<a[i]<=40:
            c+=1
            if b[i]==1:
               C+=1
        if 40<a[i]<=60:
            v+=1
            if b[i]==1:
               V+=1
    print('snr: 0(纯噪音) 模型预测准确率：'+str(Q/q)+' 此分组总样本数:'+str(q)+'\n',
          'snr: 0-15 模型预测准确率：'+str(W/w)+' 此分组总样本数:'+str(w)+'\n',
          'snr: 15-20 模型预测准确率：'+str(E/e)+' 此分组总样本数:'+str(e)+'\n',
          'snr: 20-25 模型预测准确率：'+str(R/r)+' 此分组总样本数:'+str(r)+'\n',
          'snr: 25-30 模型预测准确率：'+str(Z/z)+' 此分组总样本数:'+str(z)+'\n',
          'snr: 30-35 模型预测准确率：'+str(X/x)+' 此分组总样本数:'+str(x)+'\n',
          'snr: 35-40 模型预测准确率：'+str(C/c)+' 此分组总样本数:'+str(c)+'\n',
          'snr: 40-60 模型预测准确率：'+str(V/v)+' 此分组总样本数:'+str(v))


Model_Predictive('snr_and_acc.txt')


