import numpy as np
from functions_a3 import flatPlate, VOR2D, inv_transform_velocity_l_to_g

## Definition of Geometry
fltplt4 = flatPlate(4,1)
n = np.array([0,1])
trailing_vortex = 1.2 * fltplt4.c

## Calculation of Influence Coefficients
inf_coef = np.zeros((fltplt4.N+1,fltplt4.N+1))

for i in range(fltplt4.N): # Collocation points
    for j in range(fltplt4.N): # Quarter chord vortex elements
        ind_vel = VOR2D(1, fltplt4.cp_x[i], fltplt4.cp_z[i], fltplt4.qc_x[j], fltplt4.qc_z[j])
        inf_coef[i, j] = np.dot(ind_vel,n)
    ind_vel_last_wake = VOR2D(1, fltplt4.cp_x[i], fltplt4.cp_z[i], trailing_vortex, 0)
    inf_coef[i, -1] = np.dot(ind_vel_last_wake,n)

inf_coef[-1,:] = np.ones((1,fltplt4.N+1))
print(inf_coef)

## Calculation of Momentary RHS Vector
U_inf = 10
alpha = np.deg2rad(5)
X0_dot = -U_inf
Z0_dot = 0
rho = 1.293
l = fltplt4.c/fltplt4.N
print(l)

# t0
RHS = np.zeros((fltplt4.N + 1,1))
for i in range(fltplt4.N):
    RHS[i] = np.dot(inv_transform_velocity_l_to_g(alpha, X0_dot, Z0_dot),n)
print(RHS)

gamma = np.linalg.solve(inf_coef, RHS)

print(gamma)

delta_p = rho*U_inf*gamma/l
L = np.sum(delta_p*l*np.cos(alpha))
