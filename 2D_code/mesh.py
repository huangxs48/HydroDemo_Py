import numpy as np 

def Mesh1D(xmin, xmax, N):
	#construct the mesh
	assert(xmax > xmin)

	#cell interface
	x1f = np.linspace(xmin, xmax, Nx+1, endpoint=True)

	dx = (xmax - xmin)/N
	dxh = dx/2.
	#cell centers
	x1v = np.linspace(xmin+dxh, xmax-dxh, Nx, endpoint=True)
	
	return np.array([x1f, x1v], dtype=object)

def Mesh2D(xmin, xmax, Nx, ymin, ymax, Ny):
	assert(xmax > xmin)
	assert(ymax > ymin)

	#cell interface
	x1f = np.linspace(xmin, xmax, Nx+1, endpoint=True)

	dx = (xmax - xmin)/Nx
	dxh = dx/2.
	#cell centers
	x1v = np.linspace(xmin+dxh, xmax-dxh, Nx, endpoint=True)

	x2f = np.linspace(ymin, ymax, Ny+1, endpoint=True)

	dy = (ymax - ymin)/Ny
	dyh = dy/2.
	#cell centers
	x2v = np.linspace(ymin+dyh, ymax-dyh, Ny, endpoint=True)

	x1v_grid, x2v_grid = np.meshgrid(x1v, x2v)
	x1f_grid, x2f_grid = np.meshgrid(x1f, x2f)

	#return np.array([x1f_grid, x1v_grid, x2f_grid, x2v_grid], dtype=object)
	return x1f_grid, x1v_grid, x2f_grid, x2v_grid



	