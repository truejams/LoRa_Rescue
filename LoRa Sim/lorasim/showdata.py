import numpy as np
import matplotlib.pyplot as plt
data = np.loadtxt("C:\\Users\\grego\\Documents\\JAMES DATA\\Python Projs\\LoRa_Rescue\\LoRa Sim\\lorasim\\exp2BS3.dat")
x,y=np.loadtxt("C:\\Users\\grego\\Documents\\JAMES DATA\\Python Projs\\LoRa_Rescue\\LoRa Sim\\lorasim\\exp2BS3.dat",unpack=True) #thanks warren!
x,y =  zip(*data)
plt.plot(x, y, linewidth=2.0)
# plt.plot(*np.loadtxt("C:\\Users\\grego\\Documents\\JAMES DATA\\Python Projs\\LoRa_Rescue\\LoRa Sim\\lorasim\\exp2BS3.dat",unpack=True), linewidth=2.0)
plt.show()