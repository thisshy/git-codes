import numpy as np
import matplotlib.pyplot as plt
a = np.loadtxt('0_data.txt')

b=a[:]
c=b[:,[1,2]]
np.savetxt('H.txt',c)
np.savez('H.npy',c)
time=a[:,0]
Hhn=a[:,1]
Hht=a[:,2]
fig=plt.figure(figsize=(15,8),dpi=80)
plt.subplot(511)
plt.plot(time,Hhn,'b-',linewidth=0.4,label='hn')
plt.plot(time,Hht,'r-',linewidth=0.8,label='ht')
plt.xlim(-0.1,2.25,0.25)
plt.xlabel('time')
plt.title('Dectector H')
plt.legend(loc=1)



c=b[:,[3,4]]
np.savetxt('L.txt',c)
np.savez('L.npy',c)
time=a[:,0]
Lhn=a[:,3]
Lht=a[:,4]
plt.subplot(513)
plt.plot(time,Lhn,'b-',linewidth=0.4,label='hn')
plt.plot(time,Lht,'r-',linewidth=0.8,label='ht')
plt.xlim(-0.1,2.25,0.25)
plt.xlabel('time')
plt.title('Dectector L')
plt.legend(loc=1)




c=b[:,[5,6]]
np.savetxt('V.txt',c)
np.savez('V.npy',c)
time=a[:,0]
Vhn=a[:,5]
Vht=a[:,6]
plt.subplot(515)
plt.plot(time,Vhn,'b-',linewidth=0.4,label='hn')
plt.plot(time,Vht,'r-',linewidth=0.8,label='ht')
plt.xlim(-0.1,2.25,0.25)
plt.xlabel('time')
plt.title('Dectector V')
plt.legend(loc=1)
plt.show()

