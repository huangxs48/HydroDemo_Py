import numpy as np 

def constantrecon_x1face(rho_full, vx_full, vy_full, press_full):
    '''
    first order, just use cell center values
    
    '''

    rho_L = rho_full[:,1:-2]
    rho_R = rho_full[:,2:-1]
    vx_L = vx_full[:,1:-2]
    vx_R = vx_full[:,2:-1]
    vy_L = vy_full[:,1:-2]
    vy_R = vy_full[:,2:-1]
    press_L = press_full[:,1:-2]
    press_R = press_full[:,2:-1]
    return rho_L, rho_R, vx_L, vx_R, vy_L, vy_R, press_L, press_R

def constantrecon_x2face(rho_full, vx_full, vy_full, press_full):
    '''
    first order, just use cell center values
    
    '''

    rho_L = rho_full[1:-2,:]
    rho_R = rho_full[2:-1,:]
    vx_L = vx_full[1:-2,:]
    vx_R = vx_full[2:-1,:]
    vy_L = vy_full[1:-2,:]
    vy_R = vy_full[2:-1,:]
    press_L = press_full[1:-2,:]
    press_R = press_full[2:-1,:]
    return rho_L, rho_R, vx_L, vx_R, vy_L, vy_R, press_L, press_R


def minmod(x, y, z):
    return (1.0/4.0) * np.abs(np.sign(x) + np.sign(y)) \
            * (np.sign(x) + np.sign(z)) \
            * np.minimum(np.abs(x), np.minimum(np.abs(y), np.abs(z)))

def minmodrecon_x1face(rho_full, vx_full, vy_full, press_full):
    #index, i range from 0 - nx, il = i-nghost, iu = i+nghost
    #print(rho_full)
    nghost = 2
    il = 0
    iu = np.shape(rho_full)[1] - 1
    si = il + nghost
    ei = iu - nghost

    theta = 1.5

    # rhoL_c = rho_full[si:iu]
    # rhoL_m = rho_full[il:iu-2] # c(i-2)
    # rhoL_mm = rho_full[il+1:iu-1]# c(i-1)
    # rhoL_p = rho_full[si:iu]
    # rhoL_pp = rho_full[si+1:iu+1]

    #left state
    rhoL_c = rho_full[:,si-1:iu-1]#[il:iu-2]
    rhoL_m = rho_full[:,si-2:iu-2]
    rhoL_p = rho_full[:,si:iu]

    rho_L = rhoL_c + 0.5 * minmod(theta * (rhoL_c - rhoL_m), \
            0.5 * (rhoL_p - rhoL_m), theta * (rhoL_p - rhoL_c))
    # print(rhoL_c, rhoL_m, rhoL_p)
    # print(minmod(theta * (rhoL_c - rhoL_m), \
    #       0.5 * (rhoL_p - rhoL_m), theta * (rhoL_p - rhoL_c)))
    #print(rho_L)

    #right state
    rhoR_c = rho_full[:,si:iu]
    rhoR_m = rho_full[:,si-1:iu-1]
    rhoR_p = rho_full[:,si+1:iu+1]

    rho_R = rhoR_c - 0.5 * minmod(theta * (rhoR_c - rhoR_m), \
           0.5 * (rhoR_p - rhoL_m), theta * (rhoR_p - rhoR_c))
    # print(rhoR_c, rhoR_m, rhoR_p)
    # print(minmod(theta * (rhoR_c - rhoR_m), \
    #        0.5 * (rhoR_p - rhoL_m), theta * (rhoR_p - rhoR_c)))
    #print(rho_R)

    #left state
    vxL_c = vx_full[:,si-1:iu-1]#[il:iu-2]
    vxL_m = vx_full[:,si-2:iu-2]
    vxL_p = vx_full[:,si:iu]


    vx_L = vxL_c + 0.5 * minmod(theta * (vxL_c - vxL_m), \
            0.5 * (vxL_p - vxL_m), theta * (vxL_p - vxL_c))
    # print(vxL_c, vxL_m, vxL_p)
    # print(minmod(theta * (vxL_c - vxL_m), \
    #       0.5 * (vxL_p - vxL_m), theta * (vxL_p - vxL_c)))

    #right state
    vxR_c = vx_full[:,si:iu]
    vxR_m = vx_full[:,si-1:iu-1]
    vxR_p = vx_full[:,si+1:iu+1]

    vx_R = vxR_c - 0.5 * minmod(theta * (vxR_c - vxR_m), \
           0.5 * (vxR_p - vxL_m), theta * (vxR_p - vxR_c))
    # print(vxR_c, vxR_m, vxR_p)
    # print(minmod(theta * (vxR_c - vxR_m), \
    #       0.5 * (vxR_p - vxR_m), theta * (vxR_p - vxR_c)))

    #left state
    vyL_c = vy_full[:,si-1:iu-1]#[il:iu-2]
    vyL_m = vy_full[:,si-2:iu-2]
    vyL_p = vy_full[:,si:iu]

    vy_L = vyL_c + 0.5 * minmod(theta * (vyL_c - vyL_m), \
            0.5 * (vyL_p - vyL_m), theta * (vyL_p - vyL_c))
    # print(vxL_c, vxL_m, vxL_p)
    # print(minmod(theta * (vxL_c - vxL_m), \
    #       0.5 * (vxL_p - vxL_m), theta * (vxL_p - vxL_c)))

    #right state
    vyR_c = vy_full[:,si:iu]
    vyR_m = vy_full[:,si-1:iu-1]
    vyR_p = vy_full[:,si+1:iu+1]

    vy_R = vyR_c - 0.5 * minmod(theta * (vyR_c - vyR_m), \
           0.5 * (vyR_p - vyL_m), theta * (vyR_p - vyR_c))

    #left state
    pressL_c = press_full[:,si-1:iu-1]#[il:iu-2]
    pressL_m = press_full[:,si-2:iu-2]
    pressL_p = press_full[:,si:iu]

    press_L = pressL_c + 0.5 * minmod(theta * (pressL_c - pressL_m), \
            0.5 * (pressL_p - pressL_m), theta * (pressL_p - pressL_c))
    # print(pressL_c, pressL_m, pressL_p)
    # print(minmod(theta * (pressL_c - pressL_m), \
    #       0.5 * (pressL_p - pressL_m), theta * (pressL_p - pressL_c)))

    #right state
    pressR_c = press_full[:,si:iu]
    pressR_m = press_full[:,si-1:iu-1]
    pressR_p = press_full[:,si+1:iu+1]

    press_R = pressR_c - 0.5 * minmod(theta * (pressR_c - pressR_m), \
           0.5 * (pressR_p - pressL_m), theta * (pressR_p - pressR_c))
    # print(pressR_c)
    # print(0.5*minmod(theta * (pressR_c - pressR_m), 0.5 * (pressR_p - pressR_m), theta * (pressR_p - pressR_c)))

    return rho_L, rho_R, vx_L, vx_R, vy_L, vy_R, press_L, press_R


def minmodrecon_x2face(rho_full, vx_full, vy_full, press_full):
    #index, i range from 0 - nx, il = i-nghost, iu = i+nghost
    #print(rho_full)
    nghost = 2
    il = 0
    iu = np.shape(rho_full)[0] - 1
    si = il + nghost
    ei = iu - nghost

    theta = 1.5

    # rhoL_c = rho_full[si:iu]
    # rhoL_m = rho_full[il:iu-2] # c(i-2)
    # rhoL_mm = rho_full[il+1:iu-1]# c(i-1)
    # rhoL_p = rho_full[si:iu]
    # rhoL_pp = rho_full[si+1:iu+1]

    #left state
    rhoL_c = rho_full[si-1:iu-1,:]#[il:iu-2]
    rhoL_m = rho_full[si-2:iu-2,:]
    rhoL_p = rho_full[si:iu,:]


    rho_L = rhoL_c + 0.5 * minmod(theta * (rhoL_c - rhoL_m), \
            0.5 * (rhoL_p - rhoL_m), theta * (rhoL_p - rhoL_c))
    # print(rhoL_c, rhoL_m, rhoL_p)
    # print(minmod(theta * (rhoL_c - rhoL_m), \
    #       0.5 * (rhoL_p - rhoL_m), theta * (rhoL_p - rhoL_c)))
    # print(rho_L)

    #right state
    rhoR_c = rho_full[si:iu,:]
    rhoR_m = rho_full[si-1:iu-1,:]
    rhoR_p = rho_full[si+1:iu+1,:]

    rho_R = rhoR_c - 0.5 * minmod(theta * (rhoR_c - rhoR_m), \
           0.5 * (rhoR_p - rhoL_m), theta * (rhoR_p - rhoR_c))
    # print(rhoR_c, rhoR_m, rhoR_p)
    # print(minmod(theta * (rhoR_c - rhoR_m), \
    #        0.5 * (rhoR_p - rhoL_m), theta * (rhoR_p - rhoR_c)))
    # print(rho_R)

    #left state
    vxL_c = vx_full[si-1:iu-1,:]#[il:iu-2]
    vxL_m = vx_full[si-2:iu-2,:]
    vxL_p = vx_full[si:iu,:]


    vx_L = vxL_c + 0.5 * minmod(theta * (vxL_c - vxL_m), \
            0.5 * (vxL_p - vxL_m), theta * (vxL_p - vxL_c))
    # print(vxL_c, vxL_m, vxL_p)
    # print(minmod(theta * (vxL_c - vxL_m), \
    #       0.5 * (vxL_p - vxL_m), theta * (vxL_p - vxL_c)))

    #right state
    vxR_c = vx_full[si:iu,:]
    vxR_m = vx_full[si-1:iu-1,:]
    vxR_p = vx_full[si+1:iu+1,:]


    vx_R = vxR_c - 0.5 * minmod(theta * (vxR_c - vxR_m), \
           0.5 * (vxR_p - vxL_m), theta * (vxR_p - vxR_c))
    # print(vxR_c, vxR_m, vxR_p)
    # print(minmod(theta * (vxR_c - vxR_m), \
    #       0.5 * (vxR_p - vxR_m), theta * (vxR_p - vxR_c)))

    #left state
    vyL_c = vy_full[si-1:iu-1,:]#[il:iu-2]
    vyL_m = vy_full[si-2:iu-2,:]
    vyL_p = vy_full[si:iu,:]
    #print(vyL_m)
    #print(vyL_c)
    #print(vyL_p)

    vy_L = vyL_c + 0.5 * minmod(theta * (vyL_c - vyL_m), \
            0.5 * (vyL_p - vyL_m), theta * (vyL_p - vyL_c))
    # print(vxL_c, vxL_m, vxL_p)
    # print(minmod(theta * (vxL_c - vxL_m), \
    #       0.5 * (vxL_p - vxL_m), theta * (vxL_p - vxL_c)))

    #right state
    vyR_c = vy_full[si:iu,:]
    vyR_m = vy_full[si-1:iu-1,:]
    vyR_p = vy_full[si+1:iu+1,:]

    vy_R = vyR_c - 0.5 * minmod(theta * (vyR_c - vyR_m), \
           0.5 * (vyR_p - vyL_m), theta * (vyR_p - vyR_c))

    #left state
    pressL_c = press_full[si-1:iu-1,:]#[il:iu-2]
    pressL_m = press_full[si-2:iu-2,:]
    pressL_p = press_full[si:iu,:]


    press_L = pressL_c + 0.5 * minmod(theta * (pressL_c - pressL_m), \
            0.5 * (pressL_p - pressL_m), theta * (pressL_p - pressL_c))
    # print(pressL_c, pressL_m, pressL_p)
    # print(minmod(theta * (pressL_c - pressL_m), \
    #       0.5 * (pressL_p - pressL_m), theta * (pressL_p - pressL_c)))

    #right state
    pressR_c = press_full[si:iu,:]
    pressR_m = press_full[si-1:iu-1,:]
    pressR_p = press_full[si+1:iu+1,:]


    press_R = pressR_c - 0.5 * minmod(theta * (pressR_c - pressR_m), \
           0.5 * (pressR_p - pressL_m), theta * (pressR_p - pressR_c))
    # print(pressR_c, pressR_m, pressR_p)
    # print(minmod(theta * (pressR_c - pressR_m), \
    #       0.5 * (pressR_p - pressR_m), theta * (pressR_p - pressR_c)))


    return rho_L, rho_R, vx_L, vx_R, vy_L, vy_R, press_L, press_R

test = np.array([[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
                 [2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7],
                 [3.1, 3.2, 9.0, 8.0, 7.0, 3.6, 3.7],
                 [4.1, 4.2, 9.0, 8.0, 7.0, 4.6, 4.7],
                 [5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7],
                 [6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7]])

#minmodrecon_x2face(test, test, test, test)