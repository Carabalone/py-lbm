import numpy as np
from lbm_consts import NX, NY, Q, C, CS, W, TAU, opposite_indices, cylinder_mask


# f: [NX][NY][Q] -> ρ: [NX][NY]
# ρ = Σ f_i
def get_density(velocity_populations):
  return np.sum(velocity_populations, axis=-1)

# f: [NX][NY][Q], ρ: [NX][NY], c[D][Q] -> u: [NX][NY][D]
# u = 1/p * Σ c_i * f_i
def get_macro_velocities(velocity_populations, density):
  return np.einsum(
    "ij,xyj->xyi",
    C,
    velocity_populations
  ) / density[..., np.newaxis]

# f_eq: [NX][NY][9] (9 eq equations for each lattice)
# fI_eq = w_i * ρ * (1 + u∙c_i / c_s² + (u∙c_i)² / 2c_s⁴ - u∙u / 2c_s²)
def equilibrium(rho, u):
  f_eq = np.zeros((NX, NY, Q))
  
  for i in range(Q):
    u_dot_c = u[:, :, 0] * C[0, i] + u[:, :, 1] * C[1, i]
    u_dot_u = u[:, :, 0] ** 2 + u[:, :, 1] ** 2
    f_eq[:, :, i] = W[i] * rho * (1 + u_dot_c / (CS ** 2) +
                                  (u_dot_c ** 2) / (2 * (CS ** 4)) -
                                  u_dot_u / (2 * (CS ** 2)))
  return f_eq

def equilibrium_boundary(rho_slice, u_slice):
  NY = rho_slice.shape[0] 
  f_eq = np.zeros((NY, Q))

  for i in range(Q):
    u_dot_c = u_slice[:, 0] * C[0, i] + u_slice[:, 1] * C[1, i] 
    u_dot_u = u_slice[:, 0] ** 2 + u_slice[:, 1] ** 2 
    
    f_eq[:, i] = W[i] * rho_slice * (
      1 + u_dot_c / (CS ** 2) +
      (u_dot_c ** 2) / (2 * CS ** 4) - 
      u_dot_u / (2 * CS ** 2)
    )
  
  return f_eq

# f_i(x, t) * (1-Δt/τ) + f_eq_i * (Δt/τ)
# f: [NX][NY][9], f_eq: [NX][NY][9] -> f_new: [NX][NY][9]
def calculate_collision(f, f_eq):
  # here we assume Δt as 1
  return f * (1 - 1/TAU) + f_eq * (1/TAU)

def collide(f, rho, u):
  f_eq = equilibrium(rho, u)
  return calculate_collision(f, f_eq)

# modifies in place
def stream(f, f_new):
  # here we assume Δt as 1
  for i in range(Q):
    f[:, :, i] = np.roll(np.roll(f_new[:, :, i], C[0, i], axis=0), C[1, i], axis=1)

# f: [NX][NY][Q] -> (rho: [NX][NY], u: [NX][NY][D])
def compute_macroscopic(velocity_populations):
  density = get_density(velocity_populations)
  return (density, get_macro_velocities(velocity_populations, density=density))

# Zou/He drichlet boundary condition
# modifies in place
def apply_inflow_boundary(f, rho, u, rho_inflow, u_inflow):
  u[0, :, 0] = u_inflow
  u[0, :, 1] = 0
  rho[0, :]  = rho_inflow

  f_eq_inflow = equilibrium_boundary(rho[0, :], u[0, :, :])
  for i in range(Q):
    f[0, :, i] = f_eq_inflow[:, i] 

# Bounce-Back boundary conditions
# modifies in place
def apply_cylinder_boundary(f):
  for i in range(Q):
    f[cylinder_mask, i] = f[cylinder_mask, opposite_indices[i]]

def apply_outflow_boundary(f, rho, u):
  f[-1, :, [1, 5, 8]] = f[-2, :, [1, 5, 8]]

def apply_boundary_conditions(f, rho, u, rho_inflow, u_inflow):
  apply_inflow_boundary(f, rho, u, rho_inflow, u_inflow)
  apply_cylinder_boundary(f)
  apply_outflow_boundary(f, rho, u)