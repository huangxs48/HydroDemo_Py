import numpy as np 
from hll import HLL
from eos import cons2prim, prim2cons, timestep

def firstorderforward(rho_L, rho_R, vx_L, vx_R, \
                     press_L, press_R, gas_gamma,\
                     dt, dx,\
                     mass, momx, energy):
    #calculate HLL fluxes
    flux_mass, flux_momx, flux_energy = HLL(rho_L, rho_R, vx_L, vx_R,\
                                            press_L, press_R, gas_gamma)

    #add calculated flux to conservative
    # mass_new = addflux(1.0, mass, flux_mass, dt, dx)
    # momx_new = addflux(1.0, momx, flux_momx, dt, dx)
    # energy_new = addflux(1.0, energy, flux_energy, dt, dx)

    mass_new = mass -  (dt/dx) * (flux_mass[1:] - flux_mass[0:-1])
    momx_new = momx -  (dt/dx) * (flux_momx[1:] - flux_momx[0:-1])
    energy_new = energy -  (dt/dx) * (flux_energy[1:] - flux_energy[0:-1])

    return mass_new, momx_new, energy_new

def rk2forward(rho_L, rho_R, vx_L, vx_R, press_L, press_R, \
              gas_gamma, dt, dx,\
              mass, momx, energy):
    mass0, momx0, energy0 = mass, momx, energy

    flux_mass0, flux_momx0, flux_energy0 = HLL(rho_L, rho_R, vx_L, vx_R,\
                                            press_L, press_R, gas_gamma)

    mass1 = mass0 -  (dt/dx) * (flux_mass0[1:] - flux_mass0[0:-1])
    momx1 = momx0 -  (dt/dx) * (flux_momx0[1:] - flux_momx0[0:-1])
    energy1 = energy0 -  (dt/dx) * (flux_energy0[1:] - flux_energy0[0:-1])

    #get the updated primitives
    rho1, vx1, press1 = cons2prim(mass1, momx1, energy1, gas_gamma)

    #apply outflow boundary
    rho_L1 = np.insert(rho1, 0, rho1[0])
    rho_R1 = np.insert(rho1, -1, rho1[-1])
    vx_L1 = np.insert(vx1, 0, vx1[0])
    vx_R1 = np.insert(vx1, -1, vx1[-1])
    press_L1 = np.insert(press1, 0, press1[0])
    press_R1 = np.insert(press1, -1, press1[-1])

    #calculate L(U1)
    flux_mass1, flux_momx1, flux_energy1 = HLL(rho_L1, rho_R1, vx_L1, vx_R1,\
                                               press_L1, press_R1, gas_gamma)


    mass_new = (1.0/2.0) * mass0 + (1.0/2.0) * mass1  \
            - (1.0/2.0) * (dt/dx) * (flux_mass1[1:] - flux_mass1[0:-1])
    momx_new = (1.0/2.0) * momx0 + (1.0/2.0) * momx1 \
            - (1.0/2.0) * (dt/dx) * (flux_momx1[1:] - flux_momx1[0:-1])
    energy_new = (1.0/2.0) * energy0 + (1.0/2.0) * energy1 \
            - (1.0/2.0) * (dt/dx) * (flux_energy1[1:] - flux_energy1[0:-1])


    return mass_new, momx_new, energy_new

def rk3forward(rho_L, rho_R, vx_L, vx_R, press_L, press_R, \
              gas_gamma, dt, dx,\
              mass, momx, energy):
    mass0, momx0, energy0 = mass, momx, energy

    flux_mass0, flux_momx0, flux_energy0 = HLL(rho_L, rho_R, vx_L, vx_R,\
                                            press_L, press_R, gas_gamma)

    mass1 = mass0 -  (dt/dx) * (flux_mass0[1:] - flux_mass0[0:-1])
    momx1 = momx0 -  (dt/dx) * (flux_momx0[1:] - flux_momx0[0:-1])
    energy1 = energy0 -  (dt/dx) * (flux_energy0[1:] - flux_energy0[0:-1])

    #get the updated primitives
    rho1, vx1, press1 = cons2prim(mass1, momx1, energy1, gas_gamma)

    #apply outflow boundary
    rho_L1 = np.insert(rho1, 0, rho1[0])
    rho_R1 = np.insert(rho1, -1, rho1[-1])
    vx_L1 = np.insert(vx1, 0, vx1[0])
    vx_R1 = np.insert(vx1, -1, vx1[-1])
    press_L1 = np.insert(press1, 0, press1[0])
    press_R1 = np.insert(press1, -1, press1[-1])

    #calculate L(U1)
    flux_mass1, flux_momx1, flux_energy1 = HLL(rho_L1, rho_R1, vx_L1, vx_R1,\
                                               press_L1, press_R1, gas_gamma)


    mass2 = (3.0/4.0) * mass0 + (1.0/4.0) * mass1  \
            - (1.0/4.0) * (dt/dx) * (flux_mass1[1:] - flux_mass1[0:-1])
    momx2 = (3.0/4.0) * momx0 + (1.0/4.0) * momx1 \
            - (1.0/4.0) * (dt/dx) * (flux_momx1[1:] - flux_momx1[0:-1])
    energy2 = (3.0/4.0) * energy0 + (1.0/4.0) * energy1 \
            - (1.0/4.0) * (dt/dx) * (flux_energy1[1:] - flux_energy1[0:-1])

    #perform the third step 
    #get the updated primitives
    rho2, vx2, press2 = cons2prim(mass2, momx2, energy2, gas_gamma)

    #apply outflow boundary
    rho_L2 = np.insert(rho2, 0, rho2[0])
    rho_R2 = np.insert(rho2, -1, rho2[-1])
    vx_L2 = np.insert(vx2, 0, vx2[0])
    vx_R2 = np.insert(vx2, -1, vx2[-1])
    press_L2 = np.insert(press2, 0, press2[0])
    press_R2 = np.insert(press2, -1, press2[-1])

    #calculate L(U2)
    flux_mass2, flux_momx2, flux_energy2 = HLL(rho_L2, rho_R2, vx_L2, vx_R2,\
                                               press_L2, press_R2 , gas_gamma)

    mass_new = (1.0/3.0) * mass0 + (2.0/3.0) * mass2  \
            - (2.0/3.0) * (dt/dx) * (flux_mass2[1:] - flux_mass2[0:-1])
    momx_new = (1.0/3.0) * momx0 + (2.0/3.0) * momx2  \
            - (2.0/3.0) * (dt/dx) * (flux_momx2[1:] - flux_momx2[0:-1])
    energy_new = (1.0/3.0) * energy0 + (2.0/3.0) * energy2  \
            - (2.0/3.0) * (dt/dx) * (flux_energy2[1:] - flux_energy2[0:-1])

    return mass_new, momx_new, energy_new

