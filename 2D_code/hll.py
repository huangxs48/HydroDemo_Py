import numpy as np 
gas_gamma = 1.4

def HLLx1(rho_L, rho_R, vx_L, vx_R, vy_L, vy_R,\
	    press_L, press_R, gas_gamma):


    #inputs primitives matrix, return flux matrix
    #get moemntum and energy density
    momx_L = rho_L * vx_L
    momx_R = rho_R * vx_R
    momy_L = rho_L * vy_L
    momy_R = rho_R * vy_R
    energy_L = press_L / (gas_gamma - 1.0) + 0.5 * rho_L * (vx_L * vx_L + vy_L * vy_L)
    energy_R = press_R / (gas_gamma - 1.0) + 0.5 * rho_R * (vx_R * vx_R + vy_R * vy_R)

    #F and U in Left and Right state
    flux_rho_L = rho_L * vx_L 
    flux_rho_R = rho_R * vx_R

    flux_momx_L = rho_L * vx_L * vx_L + press_L 
    flux_momx_R = rho_R * vx_R * vx_R + press_R 

    flux_momy_L = rho_L * vy_L * vx_L
    flux_momy_R = rho_R * vy_R * vx_R

    flux_energy_L = (energy_L + press_L) * vx_L 
    flux_energy_R = (energy_R + press_R) * vx_R

    #calculate maximum and minimum signal speed
    cs_L = np.sqrt(gas_gamma * press_L / rho_L)
    cs_R = np.sqrt(gas_gamma * press_R / rho_R)
    #print(press_R)

    a_p = np.maximum(vx_L + cs_L, vx_R + cs_R)
    a_p = np.maximum(np.zeros(np.shape(vx_L)), a_p)
    a_m = np.maximum(- (vx_L - cs_L), - (vx_R - cs_R))
    a_m = np.maximum(np.zeros(np.shape(vx_R)), a_m)

    #print(a_p, a_m)

    #calculate flux
    flux_mass = (a_p * flux_rho_L + a_m * flux_rho_R \
                - a_p * a_m *(rho_R - rho_L)) / (a_p + a_m)

    flux_momx = (a_p * flux_momx_L + a_m * flux_momx_R \
                - a_p * a_m *(momx_R - momx_L)) / (a_p + a_m)

    flux_momy = (a_p * flux_momy_L + a_m * flux_momy_R \
                - a_p * a_m *(momy_R - momy_L)) / (a_p + a_m)

    flux_energy = (a_p * flux_energy_L + a_m * flux_energy_R \
                - a_p * a_m *(energy_R - energy_L)) / (a_p + a_m)

    return flux_mass, flux_momx, flux_momy, flux_energy


def HLLx2(rho_L, rho_R, vx_L, vx_R, vy_L, vy_R,\
        press_L, press_R, gas_gamma):


    #inputs primitives matrix, return flux matrix
    #get moemntum and energy density
    momx_L = rho_L * vx_L
    momx_R = rho_R * vx_R
    momy_L = rho_L * vy_L
    momy_R = rho_R * vy_R
    energy_L = press_L / (gas_gamma - 1.0) + 0.5 * rho_L * (vx_L * vx_L + vy_L * vy_L)
    energy_R = press_R / (gas_gamma - 1.0) + 0.5 * rho_R * (vx_R * vx_R + vy_R * vy_R)

    #F and U in Left and Right state
    flux_rho_L = rho_L * vy_L 
    flux_rho_R = rho_R * vy_R

    flux_momx_L = rho_L * vx_L * vy_L 
    flux_momx_R = rho_R * vx_R * vy_R 

    flux_momy_L = rho_L * vy_L * vy_L + press_L
    flux_momy_R = rho_R * vy_R * vy_R + press_R

    flux_energy_L = (energy_L + press_L) * vy_L 
    flux_energy_R = (energy_R + press_R) * vy_R

    #calculate maximum and minimum signal speed
    cs_L = np.sqrt(gas_gamma * press_L / rho_L)
    cs_R = np.sqrt(gas_gamma * press_R / rho_R)
    #print(cs_L)

    a_p = np.maximum(vy_L + cs_L, vy_R + cs_R)
    a_p = np.maximum(np.zeros(np.shape(vy_L)), a_p)
    a_m = np.maximum(- (vy_L - cs_L), - (vy_R - cs_R))
    a_m = np.maximum(np.zeros(np.shape(vy_R)), a_m)

    # print()
    # print("in HLL, ap", a_p)
    # print("in HLL, am", a_m)

    #calculate flux
    flux_mass = (a_p * flux_rho_L + a_m * flux_rho_R \
                - a_p * a_m *(rho_R - rho_L)) / (a_p + a_m)

    flux_momx = (a_p * flux_momx_L + a_m * flux_momx_R \
                - a_p * a_m *(momx_R - momx_L)) / (a_p + a_m)

    flux_momy = (a_p * flux_momy_L + a_m * flux_momy_R \
                - a_p * a_m *(momy_R - momy_L)) / (a_p + a_m)

    flux_energy = (a_p * flux_energy_L + a_m * flux_energy_R \
                - a_p * a_m *(energy_R - energy_L)) / (a_p + a_m)


    return flux_mass, flux_momx, flux_momy, flux_energy

# rhol = np.array([0.1, 1, 1, 0.1])
# vxl = np.array([1., 1., 1., 1.])
# pressl = np.array([1, 1.5, 1.5, 1])
# rhor = np.array([0.2, 1.1, 1.1, 0.1])
# vxr = np.array([1.2, 1.2, 1.2, 1.2])
# pressr = np.array([2., 2.5, 2.5, 2])

# fluxes = HLL(rhol, rhor, vxl, vxr, pressl, pressr, 5.0/3.0)


def addflux(weight, F, flux, dt, dx):
	#apply flux to conservative field F

	# print(flux)

	flux_m = flux[0:-1]
	flux_p = flux[1:]
	# print(flux_m)
	# print(flux_p)

	F += - weight * (dt/dx) * (flux_p - flux_m)

	return F

