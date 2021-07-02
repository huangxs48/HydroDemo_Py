import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd

from mesh import Mesh1D
from hll import HLL
from eos import cons2prim, prim2cons, timestep
from recon import constantrecon, minmodrecon
from timeintegration import rk3forward, rk2forward, firstorderforward

# constant
gas_gamma = 1.4 #5.0/3.0
CFL = 0.8

# domain size
xleft = 0
xright = 1.0
Nx = 128

# mesh grid
x1f = Mesh1D(xleft, xright, Nx)[0]
x1v = Mesh1D(xleft, xright, Nx)[1]

# initialize the conservatives

# initialize density profile 
rho_init = np.zeros_like(x1v)
v_init = np.zeros_like(x1v)
press_init = np.zeros_like(x1v)

rho_init = np.where(x1v<=0.5, 1.0, 0.125)
v_init = v_init 
press_init = np.where(x1v<=0.5, 1.0, 0.1)

# rho_init = np.where(x1v<=0.5, 10.0, 1.0)
# v_init = v_init 
# press_init = np.where(x1v<=0.5, 100.0, 1.0)

dx = np.gradient(x1v)
vol = dx 

tnow = 0.0
tlim = 0.2
nstep = 0
rho = rho_init 
vx = v_init 
press = press_init
mass, momx, energy = prim2cons(rho_init, v_init, press_init, gas_gamma)

fig, axs = plt.subplots(3, 1, sharex=True, figsize=(9,9))
fig.subplots_adjust(hspace=0)
ax0 = axs[0]
ax1 = axs[1]
ax2 = axs[2]

def plot(color, ls):
    ax0.plot(x1v, rho, c=color, ls=ls)
    ax1.plot(x1v, vx, c=color, ls=ls)
    ax2.plot(x1v, press, c=color, ls=ls)

    ax0.scatter(x1v, rho, c=color)
    ax1.scatter(x1v, vx, c=color)
    ax2.scatter(x1v, press, c=color)

    for ax in axs:
        ax.grid(True)


plot('k', '--')

def onestep():
    global tnow, nstep
    global mass, momx, energy
    global rho, vx, press


    mass, momx, energy = prim2cons(rho, vx, press, gas_gamma)
    dt = timestep(rho, vx, press, dx, CFL, gas_gamma)


    #apply outflow boundary

    rho_full = np.insert(rho, 0, [rho[0], rho[0]])
    rho_full = np.insert(rho_full, -1, [rho[-1], rho[-1]])
    vx_full = np.insert(vx, 0, [vx[0], vx[0]])
    vx_full = np.insert(vx_full, -1, [vx[-1], vx[-1]])
    press_full = np.insert(press, 0, [press[0], press[0]])
    press_full = np.insert(press_full, -1, [press[-1], press[-1]])

    # rho_full = np.insert(rho, 0, rho[0])
    # rho_full = np.insert(rho_full, -1, rho[-1])
    # vx_full = np.insert(vx, 0, vx[0])
    # vx_full = np.insert(vx_full, -1, vx[-1])
    # press_full = np.insert(press, 0, press[0])
    # press_full = np.insert(press_full, -1, press[-1])


    rho_L, rho_R, vx_L, vx_R, press_L, press_R = \
    constantrecon(rho_full, vx_full, press_full)

    # rho_L, rho_R, vx_L, vx_R, press_L, press_R = \
    # minmodrecon(rho_full, vx_full, press_full)

    mass, momx, energy = \
    firstorderforward(rho_L, rho_R, vx_L, vx_R,\
                     press_L, press_R, gas_gamma,\
                     dt, dx,\
                     mass, momx, energy)

    # mass, momx, energy = \
    # rk3forward(rho_L, rho_R, vx_L, vx_R, press_L, press_R, \
    #           gas_gamma, dt, dx,\
    #           mass, momx, energy)

    rho, vx, press = cons2prim(mass, momx, energy, gas_gamma)

    #update time
    tnow += dt 
    nstep += 1



def writeoutput(x1v, mass, momx, energy, csvname):
    df_data = np.array([x1v, mass, momx, energy,\
                        rho, vx, press])
    df_frame = pd.DataFrame({"x1v":df_data[0,:],
                             "mass":df_data[1,:],
                             "momx":df_data[2,:],
                             "energy":df_data[3,:],
                             "rho":df_data[4,:],
                             "vx":df_data[5,:],
                             "press":df_data[6,:]})
    df_frame.to_csv(csvname, index=False)

# for i in range(0,11):
#     onestep()
#onestep()
#onestep()

while tnow <= 0.2:
    print("t = ", tnow, ", step = ", nstep)
    onestep()

#writeoutput(x1v, mass, momx, energy, "./output/minmod_cfl08.csv")


plot('tab:blue', '--')
plt.show()



