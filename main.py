import numpy as np
from functions_a3 import flatPlate, VOR2D

## Definition of Geometry
fltplt4 = flatPlate(4,1)
n = np.array([0,1])

inf_coef = np.zeros((fltplt4.N,fltplt4.N))

for i in range(fltplt4.N):
    for j in range(fltplt4.N):
        ind_vel = VOR2D(1, fltplt4.cp_x[i], fltplt4.cp_z[i], fltplt4.qc_x[j], fltplt4.qc_z[j])
        inf_coef[i, j] = np.dot(ind_vel,n)

print(inf_coef)