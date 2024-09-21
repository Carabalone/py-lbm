import numpy as np
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

# ---------------------- Boundar Conditions ------------------------
CYLINDER_CENTER = (NX // 4, NY // 2)
CYLINDER_RADIUS = NY // 9 

X, Y = np.meshgrid(np.arange(NX), np.arange(NY), indexing='ij')
cylinder_mask = (X - CYLINDER_CENTER[0])**2 + (Y - CYLINDER_CENTER[1])**2 < CYLINDER_RADIUS**2

u_inflow = 0.1
rho_inflow = 1.0
# ------------------------------------------------------------------