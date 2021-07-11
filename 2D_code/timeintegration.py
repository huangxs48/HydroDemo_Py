import numpy as np 
from hll import HLLx1, HLLx2
from eos import cons2prim, prim2cons, timestep
from applyboundary import Outflow2D
from recon import constantrecon_x1face, constantrecon_x2face, minmodrecon_x1face, minmodrecon_x2face

def firstorderforwardx1(rho_L, rho_R, vx_L, vx_R, vy_L, vy_R,\
                        press_L, press_R, gas_gamma,\
                        dt, dx,\
                        mass, momx, momy, energy):
    #calculate HLL fluxes
    flux_mass_x1, flux_momx_x1, flux_momy_x1, flux_energy_x1 = HLLx1(rho_L, rho_R, vx_L, vx_R, vy_L, vy_R,\
                                                              press_L, press_R, gas_gamma)
  
    mass_new = mass -  (dt/dx) * (flux_mass_x1[2:-2,1:] - flux_mass_x1[2:-2,0:-1])

    momx_new = momx -  (dt/dx) * (flux_momx_x1[2:-2,1:] - flux_momx_x1[2:-2,0:-1])

    momy_new = momy -  (dt/dx) * (flux_momy_x1[2:-2,1:] - flux_momy_x1[2:-2,0:-1])
    
    energy_new = energy -  (dt/dx) * (flux_energy_x1[2:-2,1:] - flux_energy_x1[2:-2,0:-1])

    return mass_new, momx_new, momy_new, energy_new

def firstorderforwardx2(rho_L, rho_R, vx_L, vx_R, vy_L, vy_R,\
                     press_L, press_R, gas_gamma,\
                     dt, dy,\
                     mass, momx, momy, energy):
    #calculate HLL fluxes
    flux_mass_x2, flux_momx_x2, flux_momy_x2,flux_energy_x2 = HLLx2(rho_L, rho_R, vx_L, vx_R, vy_L, vy_R,\
                                                       press_L, press_R, gas_gamma)    


    mass_new = mass -  (dt/dy) * (flux_mass_x2[1:, 2:-2] - flux_mass_x2[0:-1, 2:-2])
                    
    momx_new = momx -  (dt/dy) * (flux_momx_x2[1:, 2:-2] - flux_momx_x2[0:-1, 2:-2])
                    
    momy_new = momy -  (dt/dy) * (flux_momy_x2[1:, 2:-2] - flux_momy_x2[0:-1, 2:-2])
    

    energy_new = energy -  (dt/dy) * (flux_energy_x2[1:, 2:-2] - flux_energy_x2[0:-1, 2:-2])
                        

    return mass_new, momx_new, momy_new, energy_new

test = np.array([[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
                 [2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7],
                 [3.1, 3.2, 9.0, 8.0, 7.0, 3.6, 3.7],
                 [4.1, 4.2, 9.0, 8.0, 7.0, 4.6, 4.7],
                 [5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7],
                 [6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7]])

# test2 = np.hstack((test[:,0].reshape(np.shape(test)[0],1), test))
# test2 = np.hstack((test, test[:,-1].reshape(np.shape(test)[0],1)))
# print(test2)
# test3 = np.vstack((test[0,:], test))
# print(test3)

# Ny = np.shape(test)[0]
# Nx = np.shape(test)[1]
# li = 0 
# ui = Nx
# nghost = 2
# si = li+nghost
# ei = ui-nghost

# lj = 0 
# uj = Ny 
# sj = lj+nghost
# ej = uj-nghost

# # print(test[1:6, 1:7])
# # print(test[1:6, 0:6])
# # print(test[sj-1:uj, si-1:ui])
# # print(test[sj-1:uj, si-2:ui-1])
# print(test[sj:ej, si-1:ui])
# print(test[sj:ej, si-2:ui-1])
# print()


def rk2forward(rho_full, vx_full, vy_full, press_full, \
                 gas_gamma, dt, dx, dy,\
                 mass, momx, momy, energy):

    mass0, momx0, momy0, energy0 = mass, momx, momy, energy
    rho0, vx0, vy0, press0 = cons2prim(mass0, momx0, momy0, energy0, gas_gamma)

    # rho_x1L, rho_x1R, vx_x1L, vx_x1R, vy_x1L, vy_x1R, press_x1L, press_x1R = \
    # constantrecon_x1face(rho_full, vx_full, vy_full, press_full)
    # rho_x2L, rho_x2R, vx_x2L, vx_x2R, vy_x2L, vy_x2R, press_x2L, press_x2R = \
    # constantrecon_x2face(rho_full, vx_full, vy_full, press_full)

    rho_x1L, rho_x1R, vx_x1L, vx_x1R, vy_x1L, vy_x1R, press_x1L, press_x1R = \
    minmodrecon_x1face(rho_full, vx_full, vy_full, press_full)
    rho_x2L, rho_x2R, vx_x2L, vx_x2R, vy_x2L, vy_x2R, press_x2L, press_x2R = \
    minmodrecon_x2face(rho_full, vx_full, vy_full, press_full)

    flux_mass0_x1, flux_momx0_x1, flux_momy0_x1, flux_energy0_x1 \
    = HLLx1(rho_x1L, rho_x1R, vx_x1L, vx_x1R, vy_x1L, vy_x1R, press_x1L, press_x1R, gas_gamma)

    flux_mass0_x2, flux_momx0_x2, flux_momy0_x2,flux_energy0_x2 \
    = HLLx2(rho_x2L, rho_x2R, vx_x2L, vx_x2R, vy_x2L, vy_x2R, press_x2L, press_x2R, gas_gamma) 


    mass1 = mass0 -  (dt/dx) * (flux_mass0_x1[2:-2,1:] - flux_mass0_x1[2:-2,0:-1])\
                  -  (dt/dy) * (flux_mass0_x2[1:, 2:-2] - flux_mass0_x2[0:-1, 2:-2])
    momx1 = momx0 -  (dt/dx) * (flux_momx0_x1[2:-2,1:] - flux_momx0_x1[2:-2,0:-1])\
                  -  (dt/dy) * (flux_momx0_x2[1:, 2:-2] - flux_momx0_x2[0:-1, 2:-2])
    momy1 = momy0 -  (dt/dx) * (flux_momy0_x1[2:-2,1:] - flux_momy0_x1[2:-2,0:-1])\
                  -  (dt/dy) * (flux_momy0_x2[1:, 2:-2] - flux_momy0_x2[0:-1, 2:-2])
    energy1 = energy0 -  (dt/dx) * (flux_energy0_x1[2:-2,1:] - flux_energy0_x1[2:-2,0:-1])\
                      -  (dt/dy) * (flux_energy0_x2[1:, 2:-2] - flux_energy0_x2[0:-1, 2:-2])

    #get the updated primitives
    rho1, vx1, vy1, press1 = cons2prim(mass1, momx1, momy1, energy1, gas_gamma)

    ##### do the second half step #########

    # #apply outflow boundary
    rho1_full = Outflow2D(rho1)
    vx1_full = Outflow2D(vx1)
    vy1_full = Outflow2D(vy1)
    press1_full = Outflow2D(press1)

    # #do X1
    # rho_x1L1, rho_x1R1, vx_x1L1, vx_x1R1, vy_x1L1, vy_x1R1, press_x1L1, press_x1R1 = \
    # constantrecon_x1face(rho1_full, vx1_full, vy1_full, press1_full)

    # rho_x2L1, rho_x2R1, vx_x2L1, vx_x2R1, vy_x2L1, vy_x2R1, press_x2L1, press_x2R1 = \
    # constantrecon_x2face(rho1_full, vx1_full, vy1_full, press1_full)

    rho_x1L1, rho_x1R1, vx_x1L1, vx_x1R1, vy_x1L1, vy_x1R1, press_x1L1, press_x1R1 = \
    minmodrecon_x1face(rho1_full, vx1_full, vy1_full, press1_full)

    rho_x2L1, rho_x2R1, vx_x2L1, vx_x2R1, vy_x2L1, vy_x2R1, press_x2L1, press_x2R1 = \
    minmodrecon_x2face(rho1_full, vx1_full, vy1_full, press1_full)

    #calculate L(U1)
    flux_mass1_x1, flux_momx1_x1, flux_momy1_x1, flux_energy1_x1 \
    = HLLx1(rho_x1L1, rho_x1R1, vx_x1L1, vx_x1R1, vy_x1L1, vy_x1R1, press_x1L1, press_x1R1, gas_gamma)

    flux_mass1_x2, flux_momx1_x2, flux_momy1_x2, flux_energy1_x2 \
    = HLLx2(rho_x2L1, rho_x2R1, vx_x2L1, vx_x2R1, vy_x2L1, vy_x2R1, press_x2L1, press_x2R1, gas_gamma)

    mass_new = (1.0/2.0) * mass0 + (1.0/2.0) * mass1  \
             - (1.0/2.0) * (dt/dx) * (flux_mass1_x1[2:-2,1:] - flux_mass1_x1[2:-2,0:-1]) \
             - (1.0/2.0) * (dt/dy) * (flux_mass1_x2[1:, 2:-2] - flux_mass1_x2[0:-1, 2:-2])

    momx_new = (1.0/2.0) * momx0 + (1.0/2.0) * momx1 \
             - (1.0/2.0) * (dt/dx) * (flux_momx1_x1[2:-2,1:] - flux_momx1_x1[2:-2,0:-1]) \
             - (1.0/2.0) * (dt/dy) * (flux_momx1_x2[1:, 2:-2] - flux_momx1_x2[0:-1, 2:-2])

    momy_new = (1.0/2.0) * momy0 + (1.0/2.0) * momy1 \
             - (1.0/2.0) * (dt/dx) * (flux_momy1_x1[2:-2,1:] - flux_momy1_x1[2:-2,0:-1]) \
             - (1.0/2.0) * (dt/dy) * (flux_momy1_x2[1:, 2:-2] - flux_momy1_x2[0:-1, 2:-2])

    energy_new = (1.0/2.0) * energy0 + (1.0/2.0) * energy1 \
               - (1.0/2.0) * (dt/dx) * (flux_energy1_x1[2:-2,1:] - flux_energy1_x1[2:-2,0:-1]) \
               - (1.0/2.0) * (dt/dy) * (flux_energy1_x2[1:, 2:-2] - flux_energy1_x2[0:-1, 2:-2])


    return mass_new, momx_new, momy_new, energy_new
