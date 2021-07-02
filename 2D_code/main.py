import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd

from mesh import Mesh2D
from hll import HLLx1
from eos import cons2prim, prim2cons, timestep
from recon import constantrecon_x1face, constantrecon_x2face
from timeintegration import firstorderforwardx1, firstorderforwardx2
from applyboundary import Outflow2D

# constant
gas_gamma = 1.4 #5.0/3.0
CFL = 0.3

# domain size
xleft = 0
xright = 1.0
Nx = 128
yleft = 0
yright = 1.0
Ny = 128

# mesh grid
x1f, x1v, x2f, x2v = Mesh2D(xleft, xright, Nx, yleft, yright, Ny)

# initialize the conservatives

# initialize density profile 
rho_init = np.zeros_like(x1v)
vx_init = np.zeros_like(x1v)
vy_init = np.zeros_like(x1v)
press_init = np.zeros_like(x1v)

# rho_init = np.where(x1v<=0.5, 1.0, 0.125)
# press_init = np.where(x1v<=0.5, 1.0, 0.1)
rad_init = np.sqrt((x2v-0.5)*(x2v-0.5) + (x1v-0.5)*(x1v-0.5))
rho_init = np.where(rad_init<=0.1, 1.0, 0.125)
press_init = np.where(rad_init<=0.1, 1.0, 0.1)

dx = np.gradient(x1v, axis=1)
dy = np.gradient(x2v, axis=0)

tnow = 0.0
tlim = 0.2
nstep = 0
rho = rho_init 
vx = vx_init
vy = vy_init
press = press_init
mass, momx, momy, energy \
= prim2cons(rho_init, vx_init, vy_init, press_init, gas_gamma)

def onestep():
    global tnow, nstep
    global mass, momx, momy, energy
    global rho, vx, vy, press


    mass, momx, momy, energy = prim2cons(rho, vx, vy, press, gas_gamma)
    dt = timestep(rho, vx, vy, press, dx, dy, CFL, gas_gamma)

    # #apply outflow boundary
    rho_full = Outflow2D(rho)
    vx_full = Outflow2D(vx)
    vy_full = Outflow2D(vy)
    press_full = Outflow2D(press)

    # doing integration on X1
    rho_x1L, rho_x1R, vx_x1L, vx_x1R, vy_x1L, vy_x1R, press_x1L, press_x1R = \
    constantrecon_x1face(rho_full, vx_full, vy_full, press_full)


    mass, momx, momy, energy = \
    firstorderforwardx1(rho_x1L, rho_x1R, vx_x1L, vx_x1R, vy_x1L, vy_x1R,\
                     press_x1L, press_x1R, gas_gamma,\
                     dt, dx, \
                     mass, momx, momy, energy)

    rho_x2L, rho_x2R, vx_x2L, vx_x2R, vy_x2L, vy_x2R, press_x2L, press_x2R = \
    constantrecon_x2face(rho_full, vx_full, vy_full, press_full)

    mass, momx, momy, energy = \
    firstorderforwardx2(rho_x2L, rho_x2R, vx_x2L, vx_x2R, vy_x2L, vy_x2R,\
                     press_x2L, press_x2R, gas_gamma,\
                     dt, dy, \
                     mass, momx, momy, energy)


    rho, vx, vy, press = cons2prim(mass, momx, momy, energy, gas_gamma)

    # #update time
    tnow += dt 
    nstep += 1



def writeoutput(x1v, x2v, rho, vx, vy, press, framename):
    np.savez(framename, \
             x1v = x1v,
             x2v = x2v,
             rho = rho,
             vx = vx,
             vy =vy, 
             press = press)

# for i in range(0,50):
#     onestep()
# onestep()
# onestep()


while tnow <= 0.1:
    print("t = ", tnow, ", step = ", nstep)
    onestep()

plt.imshow(rho)
plt.show()

#writeoutput(x1v, x2v, rho, vx, vy, press, "./output/sodx2")





