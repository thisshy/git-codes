import numpy as np
import matplotlib.pyplot as plt

fs = 10
ts = 1 / fs
t = np.arange(-5, 5, ts)  # 生成时间序列，采样间隔0.1s
k = np.arange(t.size)  # DFT的自变量
N = t.size  # DFT的点数量
x = np.zeros_like(t)  # 生成一个与t相同结构，内容为0的np.array
x[40:50] = 1  # 设置信号的方波，范围是40-60
x[50:60] = -1
y = np.fft.fft(x) # 先np.fft.fft进行fft计算，这时的y是【0,fs】频率相对应的
# 调用np.fft.fftshift将y变为与频率范围【-fs/2,fs/2】对应，就是将大于fs/2的部分放到
# -fs/2到0之间,然后绘图的时候将用频率是f=(k*fs/N-fs/2),将频率变为【-fs/2,fs/2】之间
yf = np.abs(y)  # 计算频率域的振幅
print(y.shape)
# yf = np.angle(y)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文乱码处理
plt.rcParams['axes.unicode_minus'] = False
plt.subplot(311)
plt.plot(t, x)
plt.title('原始方波信号')
plt.legend(('采样频率fs=10'))
plt.subplot(312)
plt.title('方波信号经过fft变换后的频谱')
f = fs * k / N - fs / 2  # 计算频率
plt.plot(f, y)
iy = np.real(np.fft.ifft(y)) # 注意这里进行ifft的y要是fft计算出的y经过np.fft.fftshift的
print(iy)
# 否则会显示错误的结果,y是没有经过np.abs的，y是个复数，计算结果iy是个实数
plt.subplot(313)
plt.title('方波fft的ifft')
plt.plot(t, iy)
plt.tight_layout()  # 显示完整图片
plt.show()

