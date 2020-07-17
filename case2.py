import numpy as np
import matplotlib.pyplot as plt
from functions_a3 import *

## Definition of Geometry
fltplt4 = flatPlate(10,1)
n = np.array([0,1])
trailing_vortex = 1.2 * fltplt4.c

t_end = 5 # sec
dt = 0.1 # sec

U_inf = 10
X0_dot = -U_inf
Z0_dot = 0
rho = 1.293
l = fltplt4.c/fltplt4.N
t_array = np.arange(0, t_end+dt, dt)

ks = np.array([0.1])
average_cl = np.array([])

for k in ks:
    # alpha_array = np.full(len(t_array), np.deg2rad(alpha))
    alpha_array = np.deg2rad(5) + np.deg2rad(5)*np.sin(k*2*U_inf/fltplt4.c*t_array)
    X_wake = np.zeros(len(t_array)-1)
    Z_wake = np.zeros(len(t_array)-1)
    X_TE = np.zeros(len(t_array)) # X location trailing edge
    Z_TE = np.zeros(len(t_array)) # Z location trailing edge
    X_LE = np.zeros(len(t_array)) # X location leading edge
    Z_LE = np.zeros(len(t_array)) # Z location leading edge
    Gamma_wake = np.zeros(len(t_array)-1)

    p_dist_t = np.zeros((len(t_array), fltplt4.N+len(t_array)-1))
    cl_t = np.zeros(len(t_array))

    for i_t, t in enumerate(t_array):
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
                
                RHS[i] = -1*np.dot(inv_transform_velocity_l_to_g(alpha_array[i_t], X0_dot, Z0_dot),n)

            gamma = np.linalg.solve(inf_coef, RHS)

            # compute quantities of interest
            delta_p = rho*U_inf*gamma/l
            L = np.sum(delta_p*l*np.cos(alpha_array[i_t]))
            cl_t[i_t] = L / (0.5 * rho * U_inf**2 * fltplt4.c)

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
                RHS[i] = -1*np.dot(inv_transform_velocity_l_to_g(alpha_array[i_t], X0_dot, Z0_dot),n)
            RHS[-1] = np.sum(gamma[:fltplt4.N])
            #print(RHS)

            gamma = np.linalg.solve(inf_coef, RHS)
            
            Gamma_wake[i_t-1] = gamma[-1]
            # print(transform_local_to_global(trailing_vortex,0,alpha,X0_dot*t,Z0_dot*t))
            X_wake[i_t-1] = transform_local_to_global(trailing_vortex,0,alpha_array[i_t],X0_dot*t,Z0_dot*t)[0]
            Z_wake[i_t-1] = transform_local_to_global(trailing_vortex,0,alpha_array[i_t],X0_dot*t,Z0_dot*t)[1]
            induced_wake =  np.zeros((2,len(t_array)-1))

            for i in range(i_t-1): # Collocation points
                for j in range(fltplt4.N): # Quarter chord vortex elements
                    # print(transform_local_to_global(fltplt4.qc_x[j],fltplt4.qc_z[j],alpha,X0_dot*t,Z0_dot*t))
                    airfoil_X, airfoil_Z = transform_local_to_global(fltplt4.qc_x[j],fltplt4.qc_z[j],alpha_array[i_t],X0_dot*t,Z0_dot*t)
                    induced_wake[:,i] += VOR2D(gamma[j], X_wake[i], Z_wake[i], airfoil_X, airfoil_Z) # Airfoil induced on wake
                for k in range(i_t-1):
                    if k == i:
                        continue
                    induced_wake[:,i] += VOR2D(Gamma_wake[k], X_wake[i], Z_wake[i], X_wake[k], Z_wake[k]) # Self induced
            # print(induced_wake)
            X_wake += induced_wake[0,:] * dt
            Z_wake += induced_wake[1,:] * dt
            #print(gamma)

            # compute quantities of interest (cl with gamma over airfoil only)
            delta_p = rho*U_inf*gamma[:fltplt4.N]/l
            L = np.sum(delta_p*l*np.cos(alpha_array[i_t]))
            cl_t[i_t] = L / (0.5 * rho * U_inf**2 * fltplt4.c)
            p_dist_t[i_t, :] = rho*U_inf/l*np.hstack((np.squeeze(gamma[:fltplt4.N]), Gamma_wake))
        
        X_LE[i_t], Z_LE[i_t] = transform_local_to_global(0,0,alpha_array[i_t],X0_dot*t,Z0_dot*t)
        X_TE[i_t], Z_TE[i_t] = transform_local_to_global(fltplt4.x[-1],0,alpha_array[i_t],X0_dot*t,Z0_dot*t)

fig, ax = plt.subplots()
ax.plot(X_LE,Z_LE, '.g')
ax.plot(X_TE,Z_TE, '.r')
ax.plot(X_wake, Z_wake, 'x')
ax.plot(X_wake, Z_wake, 'x')

fig, ax = plt.subplots(2,1)
ax[0].plot(t_array, cl_t)
for i in range(p_dist_t.shape[0]):
    ax[1].plot(np.hstack((fltplt4.cp_x, X_wake)), p_dist_t[i], "x", label=str(i))
ax[1].legend()

plt.show()

# fig1, ax1 = plt.subplots()
# ax1.plot(alphas, average_cl, 'b--')
# ax1.plot(alphas, np.deg2rad(alphas)*2*np.pi, 'k--')
# ax1.set_xlabel(r"$\alpha$ [deg]")
# ax1.set_ylabel(r"$C_l$ [-]")
# plt.show()