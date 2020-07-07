import numpy as np
import matplotlib.pyplot as plt
from functions_a3 import *

## Definition of Geometry
fltplt4 = flatPlate(4,1)
n = np.array([0,1])
trailing_vortex = 1.2 * fltplt4.c

t_end = 1 # sec
dt = 0.01 # sec

U_inf = 10
alpha = np.deg2rad(5)
X0_dot = -U_inf
Z0_dot = 0
rho = 1.293
l = fltplt4.c/fltplt4.N

t_array = np.arange(0, t_end+dt, dt)

X_wake = np.zeros(len(t_array))
Z_wake = np.zeros(len(t_array))
Gamma_wake = np.zeros(len(t_array))
i_t = 0

for t in t_array:
    # print("t =",t)
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
        Gamma_wake[i_t] = gamma[-1]
        # print(transform_local_to_global(trailing_vortex,0,alpha,X0_dot*t,Z0_dot*t))
        X_wake[i_t] = transform_local_to_global(trailing_vortex,0,alpha,X0_dot*t,Z0_dot*t)[0]
        Z_wake[i_t] = transform_local_to_global(trailing_vortex,0,alpha,X0_dot*t,Z0_dot*t)[1]
        induced_wake =  np.zeros((2,len(t_array)))

        for i in range(i_t): # Collocation points
            for j in range(fltplt4.N): # Quarter chord vortex elements
                # print(transform_local_to_global(fltplt4.qc_x[j],fltplt4.qc_z[j],alpha,X0_dot*t,Z0_dot*t))
                airfoil_X, airfoil_Z = transform_local_to_global(fltplt4.qc_x[j],fltplt4.qc_z[j],alpha,X0_dot*t,Z0_dot*t)
                induced_wake[:,i] += VOR2D(gamma[j], X_wake[i], Z_wake[i], airfoil_X, airfoil_Z) # Airfoil induced on wake
            for k in range(i_t):
                if k == i:
                    continue
                induced_wake[:,i] += VOR2D(Gamma_wake[k], X_wake[i], Z_wake[i], X_wake[k], Z_wake[k]) # Self induced
        # print(induced_wake)
        X_wake += induced_wake[0,:] * dt
        Z_wake += induced_wake[1,:] * dt
        #print(gamma)
    i_t += 1
    # delta_p = rho*U_inf*gamma/l
    # L = np.sum(delta_p*l*np.cos(alpha))
    # cl = L / (0.5 * rho * U_inf**2 * fltplt4.c)
    # print("cl =",cl)
plt.figure()
plt.plot(X_wake, Z_wake, '.')
plt.plot(X_wake, Z_wake, '.')
plt.show()
