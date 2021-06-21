import numpy as np 

def Mesh1D(xmin, xmax, N):
	#construct the mesh
	assert(xmax > xmin)

	#cell interface
	x1f = np.linspace(xmin, xmax, N+1, endpoint=True)

	dx = (xmax - xmin)/N
	dxh = dx/2.
	#cell centers
	x1v = np.linspace(xmin+dxh, xmax-dxh, N, endpoint=True)
	
	return np.array([x1f, x1v], dtype=object)

