import numpy as np 

def Outflow2D(arr):
    arr_full = np.vstack((arr, [arr[-1,:], arr[-1,:]]))
    arr_full = np.vstack(([arr_full[0,:], arr_full[0,:]], arr_full))
    arrcoll = arr_full[:,0].reshape(len(arr_full[:,0]), 1)
    arr_full = np.hstack((arrcoll, arr_full))
    arr_full = np.hstack((arrcoll, arr_full))
    arrcolr = arr_full[:,-1].reshape(len(arr_full[:,-1]), 1)
    arr_full = np.hstack((arr_full, arrcolr))
    arr_full = np.hstack((arr_full, arrcolr))
    return arr_full

def Periodic2D(arr):
    arr_full = np.vstack((arr, [arr[0,:], arr[0,:]]))
    arr_full = np.vstack(([arr_full[-1,:], arr_full[-1,:]], arr_full))
    arrcoll = arr_full[:,-1].reshape(len(arr_full[:,-1]), 1)
    arr_full = np.hstack((arrcoll, arr_full))
    arr_full = np.hstack((arrcoll, arr_full))
    arrcolr = arr_full[:,0].reshape(len(arr_full[:,0]), 1)
    arr_full = np.hstack((arr_full, arrcolr))
    arr_full = np.hstack((arr_full, arrcolr))
    return arr_full