import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 

x1d = pd.read_csv("sod1d_cfl08.csv")
df = np.load('sodx2.npz')

rho = df['rho']

plt.plot(x1d.x1v, x1d.vx)
#plt.plot(df['x1v'][50,:], df['vx'][50,:])
plt.plot(df['x2v'][:,50], df['vy'][:,50])
plt.show()