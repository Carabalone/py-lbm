import numpy as np
from numpy.linalg import inv, diag
# each lattice will have this structure with discrete velocities:
#  
#   6  2  5   
#    \ | /        
#   3- 0 -1
#    / | \     
#   7  4  8           

D = 2
Q = 9
STEPS = 10_000

NX = 400
NY = 50

C = np.array([
  [0, 1, 0, -1, 0, 1, -1, -1, 1],
  [0, 0, 1, 0, -1, 1, 1, -1, -1]
])

W = np.array([
  4/9,
  1/9, 1/9, 1/9, 1/9, 
  1/36, 1/36, 1/36, 1/36 
])

# speed of sound in lattice units
CS = 1 / np.sqrt(3)

TAU = 0.55

indices = np.array(range(9))
opposite_indices = np.array([
  0, 3, 4, 1, 2, 7, 8, 5, 6
])

# ---------------------- MRT ---------------------------

# Gram-Schmidt Transformation Matrix
# Lattice Boltzmann method principles and practice pg. 420 eq 10.30
M = np.array([
    [ 1,  1,  1,  1,  1,  1,  1,  1,  1], # ρ
    [-4, -1, -1, -1, -1,  2,  2,  2,  2], # e
    [ 4, -2, -2, -2, -2,  1,  1,  1,  1], # e
    [ 0,  1,  0, -1,  0,  1, -1, -1,  1], # jx
    [ 0, -2,  0,  2,  0,  1, -1, -1,  1], # qx
    [ 0,  0,  1,  0, -1,  1,  1, -1, -1], # jy
    [ 0,  0, -2,  0,  2,  1,  1, -1, -1], # qy
    [ 0,  1, -1,  1, -1,  0,  0,  0,  0], # pxx
    [ 0,  0,  0,  0,  0,  1, -1,  1, -1], # pyy
])

M_inverse = np.linalg.inv(M)

omega_e   = 1.0    # ωₑ
omega_eps = 1.0    # ω_ε
omega_q   = 1.0    # ω_q
omega_nu  = 1/0.55    # ω_ν (1 / tau)

# Gram-Schmidt Relaxation Matrix
G = np.diag([0, omega_e, omega_eps, 0, omega_q, 0, omega_q, omega_nu, omega_nu])
# ---------------------- Boundary Conditions ------------------------
CYLINDER_CENTER = (NX // 4, NY // 2)
CYLINDER_RADIUS = NY // 9 

X, Y = np.meshgrid(np.arange(NX), np.arange(NY), indexing='ij')
cylinder_mask = (X - CYLINDER_CENTER[0])**2 + (Y - CYLINDER_CENTER[1])**2 < CYLINDER_RADIUS**2

u_inflow = 0.1
rho_inflow = 1.0
# ------------------------------------------------------------------
