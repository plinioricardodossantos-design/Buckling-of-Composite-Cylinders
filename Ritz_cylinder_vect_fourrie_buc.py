import numpy as np
from scipy.linalg import eig
from numpy.polynomial.legendre import leggauss
from prop2 import prop
from lgwt import lgwt
from gaussian_quadrature import gaussian_quadrature
from abd import abd
from evaluate_KF import evaluate_KF
from evaluate_KGz import evaluate_KGz
from evaluate_Ke import evaluate_Ke
from Plot_disp import plot_disp
import matplotlib.pyplot as plt
from plot_buckling_mode import plot_cylinder_buckling_mode
from plot_contour import plot_cylinder_buckling_contour

# Parameters
nx = 15
ny = 15
nt = 2 * nx * ny
NG = 130

# Material properties
E1, E2, v12, v21, G12, G13, G23, teta_deg, hp, L, Rc = prop(2) # check the function/file "prop.py". Use "1" for user defined
print(E1)
Np = len(teta_deg)

#range_x = np.arange(1, nx + 1).T
range_x = np.arange(1, nx + 1).reshape(-1, 1)
range_y = np.arange(1, ny + 1).reshape(1, -1) 

# Gauss quadrature
xxg, wi = gaussian_quadrature(NG, 0, L)
yyg, wj = gaussian_quadrature(NG, 0, 2 * np.pi)

xi, yi = np.meshgrid(xxg, yyg)

xg = xi.flatten()
yg = yi.flatten()

wi2 = np.repeat(wi, NG)
wj2 = np.tile(wj, NG)

wij = wi2 * wj2

# ABD matrix
Ap, Bp, Dp, As = abd(E1, E2, v12, G12, G13, G23, Np, teta_deg, hp)

C = np.block([
    [Ap, Bp, np.zeros((3, 2))],
    [Bp, Dp, np.zeros((3, 2))],
    [np.zeros((2, 3)), np.zeros((2, 3)), As]])

# Global matrices
K = np.zeros((5 * nt, 5 * nt)) 
KGz = np.zeros((5 * nt, 5 * nt))
Ke = np.zeros((5 * nt, 5 * nt))

for ii in range(NG * NG):

    I_K, I_F = evaluate_KF(xg[ii], yg[ii], C, L, Rc, range_x, range_y)

    K += I_K * wij[ii]

    I_KGz = evaluate_KGz(xg[ii], yg[ii], L, Rc, range_x, range_y)

    KGz += I_KGz * wij[ii]

    I_Ke = evaluate_Ke(xg[ii], yg[ii], L, Rc, range_x, range_y) 

    Ke += I_Ke * wij[ii]

Ktot = K + Ke

# Eigenvalue problem
eigvals, eigvecs = eig(Ktot, KGz)

eigvals = np.real(eigvals)

# Keep only finite eigenvalues
mask = np.isfinite(eigvals)
eigvals = eigvals[mask]
eigvecs = eigvecs[:, mask]

# Sort eigenvalues and eigenvectors together
idx = np.argsort(eigvals)
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

Nz_cr = eigvals[0]
P_cr = 2 * np.pi * Rc * Nz_cr / 1000
#pressure_cr = -2*Rc*Nz_cr
pressure_cr = -2*Nz_cr/Rc

#print("N_cr [N/mm]:", Nz_cr)
print("P_cr [kN]:", P_cr)
print("pressure_cr [MPa]:", pressure_cr)
#print("pressure_cr_paper [MPa]:", pressure_cr)

# First buckling mode
qq = np.real(eigvecs[:, 0])

# Split displacement DOFs
qqu = qq[:nt]
qqv = qq[nt:2*nt]
qqw = qq[2*nt:3*nt]

# Compute buckling displacements
'''u_buc = Nu @ qqu
v_buc = Nv @ qqv
w_buc = Nw @ qqw'''

#plot_disp(Nu,Nv,Nw,L,Rc) 

W = plot_cylinder_buckling_contour(qqw, range_x, range_y, L, Rc)
plot_cylinder_buckling_mode(qqw, range_x, range_y, L, Rc)
