import numpy as np
from functions_a3 import flatPlate, VOR2D, inv_transform_velocity_l_to_g

## Definition of Geometry
fltplt4 = flatPlate(4,1)
n = np.array([0,1])
trailing_vortex = 1.2 * fltplt4.c

t_end = 0.2 # sec
dt = 0.1 # sec

U_inf = 10
alpha = np.deg2rad(5)
X0_dot = -U_inf
Z0_dot = 0
rho = 1.293
l = fltplt4.c/fltplt4.N

t_array = np.arange(0, t_end+dt, dt)

for t in t_array:
    print("t =",t)
    if t == 0:
        # no wake yet, calculation as if steady

        ## Calculation of Influence Coefficients and RHS
        inf_coef = np.zeros((fltplt4.N,fltplt4.N))
        RHS = np.zeros((fltplt4.N,1))

        for i in range(fltplt4.N):
            # Collocation points
            
            for j in range(fltplt4.N):
                # Quarter chord vortex elements
                ind_vel = VOR2D(1, fltplt4.cp_x[i], fltplt4.cp_z[i], fltplt4.qc_x[j], fltplt4.qc_z[j])
                inf_coef[i, j] = np.dot(ind_vel,n)
            
            RHS[i] = -np.dot(inv_transform_velocity_l_to_g(alpha, X0_dot, Z0_dot),n)

        gamma = np.linalg.solve(inf_coef, RHS)

    else:
        ## Calculation of Influence Coefficients
        inf_coef = np.zeros((fltplt4.N+1,fltplt4.N+1))

        for i in range(fltplt4.N): # Collocation points
            for j in range(fltplt4.N): # Quarter chord vortex elements
                ind_vel = VOR2D(1, fltplt4.cp_x[i], fltplt4.cp_z[i], fltplt4.qc_x[j], fltplt4.qc_z[j])
                inf_coef[i, j] = np.dot(ind_vel,n)
            ind_vel_last_wake = VOR2D(1, fltplt4.cp_x[i], fltplt4.cp_z[i], trailing_vortex, 0)
            inf_coef[i, -1] = np.dot(ind_vel_last_wake,n)

        inf_coef[-1,:] = np.ones((1,fltplt4.N+1)) # Kelvin condition
        #print(inf_coef)

        ## Calculation of Momentary RHS Vector
        RHS = np.zeros((fltplt4.N+1,1))
        for i in range(fltplt4.N):
            RHS[i] = -np.dot(inv_transform_velocity_l_to_g(alpha, X0_dot, Z0_dot),n)
        RHS[-1] = np.sum(gamma[:fltplt4.N])
        #print(RHS)

        gamma = np.linalg.solve(inf_coef, RHS)

        #print(gamma)

    delta_p = rho*U_inf*gamma/l
    L = np.sum(delta_p*l*np.cos(alpha))
    cl = L / (0.5 * rho * U_inf**2 * fltplt4.c)
    print("cl =",cl)