import pandas as pd 
import matplotlib.pyplot as plt 

fig, axs = plt.subplots(3, 1, sharex=True, figsize=(9,9))
fig.subplots_adjust(hspace=0)
ax0 = axs[0]
ax1 = axs[1]
ax2 = axs[2]

def plot(ax0, ax1, ax2, csv, color):
	data = pd.read_csv(csv)

	ax0.plot(data.x1v, data.rho, color=color)
	ax1.plot(data.x1v, data.vx, color=color)
	ax2.plot(data.x1v, data.press, color=color)

	ax0.scatter(data.x1v, data.rho, color=color)
	ax1.scatter(data.x1v, data.vx, color=color)
	ax2.scatter(data.x1v, data.press, color=color)

plot(ax0, ax1, ax2, "constrecon_cfl08.csv", 'tab:blue')
plot(ax0, ax1, ax2, "minmod_cfl08.csv", 'tab:red')

plt.show()