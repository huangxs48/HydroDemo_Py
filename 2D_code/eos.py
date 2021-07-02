import numpy as np 

rho_floor = 1.0e-4 
press_floor = 1.0e-5

#assuming ideal gas e.o.s
def cons2prim(mass, momx, momy, energy, gas_gamma):
    #all inputs are matrix
    rho = mass 
    vx = momx / rho 
    vy = momy / rho
    press = (energy \
             - 0.5 * rho * (vx * vx + vy * vy)) * (gas_gamma - 1.) 

    # rho = np.maximum(rho, rho_floor)
    # press = np.maximum(press, press_floor)

    return rho, vx, vy, press

def prim2cons(rho, vx, vy, press, gas_gamma):
    #all inputs are matrix
    mass = rho 
    momx = rho * vx  
    momy = rho * vy
    energy = press / (gas_gamma - 1.) + 0.5 * rho * (vx * vx + vy * vy) #*vol
    e_floor = press_floor / (gas_gamma - 1.) + 0.5 * rho * (vx * vx + vy * vy) #*vol

    # mass = np.maximum(rho_floor, mass)
    # energy = np.maximum(energy, e_floor)

    return mass, momx, momy, energy


def timestep(rho, vx, vy, press, dx, dy, CFL, gas_gamma):
    #inputs are matrix
    cs = np.sqrt(gas_gamma * press / rho)
    vmx = np.maximum((cs + vx), (vx - cs))
    vmy = np.maximum((cs + vy), (vy - cs))
    dtx = CFL * dx / vmx
    dty = CFL * dy / vmy
    return np.minimum(np.min(dtx), np.min(dty))




