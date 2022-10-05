import numpy as np
import matplotlib.pyplot as plt
data=np.loadtxt('snr_and_acc.txt')
#按snr数据分成各个区间
a1=[d for d in data if d[0]==0]
a2=[d for d in data if d[0]>0 and d[0]<15 ]
a3=[d for d in data if d[0]>15 and d[0]<20]
a4=[d for d in data if d[0]>20 and d[0]<25 ]
a5=[d for d in data if d[0]>25 and d[0]<30 ]
a6=[d for d in data if d[0]>30 and d[0]<35 ]
a7=[d for d in data if d[0]>35 and d[0]<40 ]
a8=[d for d in data if d[0]>40 and d[0]<60 ]

#统计每个区间匹配成功的个数
b1=[i for i in a1 if i[1] ==1]
b2=[i for i in a2 if i[1] ==1]
b3=[i for i in a3 if i[1] ==1]
b4=[i for i in a4 if i[1] ==1]
b5=[i for i in a5 if i[1] ==1]
b6=[i for i in a6 if i[1] ==1]
b7=[i for i in a7 if i[1] ==1]
b8=[i for i in a8 if i[1] ==1]
#输出结果
print("snr: 0(纯噪音)模型预测准确率："+str("%.2f"%(len(b1)/len(a1)))+" 此分组总样本数:"+str(len(a1)))
print("snr: 0-15     模型预测准确率："+str("%.2f"%(len(b2)/len(a2)))+" 此分组总样本数:"+str(len(a2)))
print("snr: 15-20    模型预测准确率："+str("%.2f"%(len(b3)/len(a3)))+" 此分组总样本数:"+str(len(a3)))
print("snr: 20-25    模型预测准确率："+str("%.2f"%(len(b4)/len(a4)))+" 此分组总样本数:"+str(len(a4)))
print("snr: 25-30    模型预测准确率："+str("%.2f"%(len(b5)/len(a5)))+" 此分组总样本数:"+str(len(a5)))
print("snr: 30-35    模型预测准确率："+str("%.2f"%(len(b6)/len(a6)))+" 此分组总样本数:"+str(len(a6)))
print("snr: 35-40    模型预测准确率："+str("%.2f"%(len(b7)/len(a7)))+" 此分组总样本数:"+str(len(a7)))
print("snr: 40-60    模型预测准确率："+str("%.2f"%(len(b8)/len(a8)))+" 此分组总样本数:"+str(len(a8)))
