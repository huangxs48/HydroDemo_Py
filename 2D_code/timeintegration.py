import numpy as np 
from hll import HLLx1, HLLx2
from eos import cons2prim, prim2cons, timestep

def firstorderforwardx1(rho_L, rho_R, vx_L, vx_R, vy_L, vy_R,\
                     press_L, press_R, gas_gamma,\
                     dt, dx,\
                     mass, momx, momy, energy):
    #calculate HLL fluxes
    flux_mass_x1, flux_momx_x1, flux_momy_x1,flux_energy_x1 = HLLx1(rho_L, rho_R, vx_L, vx_R, vy_L, vy_R,\
                                                              press_L, press_R, gas_gamma)
  

    # print(np.shape(mass), np.shape(flux_mass_x1), np.shape(flux_m
    # print(np.shape(flux_mass_x1))
    # print(flux_mass_x1)
    # print(flux_mass_x1[1:-3, 1:])
    # print(flux_mass_x1[3:-1, 1:])

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

# test = np.array([[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
#                  [2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7],
#                  [3.1, 3.2, 9.0, 8.0, 7.0, 3.6, 3.7],
#                  [4.1, 4.2, 9.0, 8.0, 7.0, 4.6, 4.7],
#                  [5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7],
#                  [6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7]])


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







