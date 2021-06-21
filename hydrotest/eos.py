import numpy as np 

rho_floor = 1.0e-4 
press_floor = 1.0e-5

#assuming ideal gas e.o.s
def cons2prim(mass, momx, energy, gas_gamma):
    #all inputs are matrix
    rho = mass #/ vol 
    vx = momx / rho #/ vol
    press = (energy #/ vol \
             - 0.5 * rho * vx * vx) * (gas_gamma - 1.) 

    # rho = np.maximum(rho, rho_floor)
    # press = np.maximum(press, press_floor)

    return rho, vx, press

def prim2cons(rho, vx, press, gas_gamma):
    #all inputs are matrix
    mass = rho #* vol 
    momx = rho * vx #* vol 
    energy = press / (gas_gamma - 1.) + 0.5 * rho * vx * vx #*vol
    e_floor = press_floor / (gas_gamma - 1.) + 0.5 * rho * vx * vx #*vol

    # mass = np.maximum(rho_floor, mass)
    # energy = np.maximum(energy, e_floor)

    return mass, momx, energy


def timestep(rho, vx, press, dx, CFL, gas_gamma):
    #inputs are matrix
    cs = np.sqrt(gas_gamma * press / rho)
    dt = CFL * dx / np.maximum((cs + vx), (vx-cs))
    return np.min(dt)




