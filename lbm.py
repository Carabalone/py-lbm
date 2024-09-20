import numpy as np
from lbm_funcs import *

# -------- Main Objectives ------------
# this is a D2Q9 Lattice Boltzmann simulation
# the lattice boltzmann equation can be written as
# f_i(x + Δt*c_i, t + Δt) = f(x, t) + Ω_i
# using the BGK collision operator for collision this transforms to:

# f_i(x + Δt*c_i, t + Δt) = f(x, t) - Δt / τ * (f_i - f_eq_i),
# where τ is the (single) relaxation time.

# this process is divided into two parts, the first being collision / relaxation,

# the first step is to calculate:
# Ω_i = - 1 / τ * (f_i - f_eq_i) = f_i(x, t) * (1-Δt/τ) + f_eq_i * (Δt/τ)
# (the latter formula is more efficient)
# for this we have to calculate f_eq_i with the following formula:
# f_eq_i = w_i * ρ * (1 + u∙c_i / c_s² + (u∙c_i)² / 2c_s⁴ - u∙u / 2c_s²)
# where w_i are the weights, ρ is the macroscopic density, u is the macroscopic speed, c_s is the speed
# of sound in lattice units and c_i is the discrete velocity for what we are calculating.

# remember that each lattice has 9 f_i (velocity populations) and 9 f_eq_i (equilibrium velocity populations)
# then we get the value of f_i_new after collision: 
# f_i_new(x, t) = f(x, t) + Ω_i
# Notice that this is a local operation, and we do *not* substitute current f_i with f_i_new
# this will be done in the streaming step since we will propagate changes, not change them locally.

# after colision we do streaming:
# f_i(x, t) = f_i_new(x,t)

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

NX = 300
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

TAU = 0.6

indices = np.array(range(9))
opposite_indices = np.array([
  0, 3, 4, 1, 2, 7, 8, 5, 6
])

# --------- Initial Conditions of Macroscopic Quantities ----------
# ρ: [NX][NY] (1 scalar for each lattice)
rho  = np.ones((NX, NY))
# u: [NX][NY][2] (1 2D vector for each lattice)
u    = np.zeros((NX, NY, D))

# f: [NX][NY][9] (these are all the fs (velocity populations))
f     = equilibrium(rho, u)
f_new = equilibrium(rho, u)
# this works like double buffering in CG. f_new will hold the new values of f until we do streaming,
# this prevents bugs because we may overwrite values we still want to get.

# -----------------------------------------------------------------