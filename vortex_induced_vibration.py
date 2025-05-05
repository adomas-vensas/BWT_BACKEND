# This example simulates the vortex-induced vibration of a circular cylinder
# in a 2D flow using the Immersed Boundary Method (IBM) coupled with the
# Lattice Boltzmann Method (LBM). The cylinder is placed at the center of the
# domain and is free to move in both directions with spring constraints. 
# The flow is driven by a constant velocity U0 in the x direction. 

import os
import math
import jax
import jax.numpy as jnp
from tqdm import tqdm
from calculations import dyn, ib, lbm, mrt, post
import matplotlib.pyplot as plt
import matplotlib as mpl

# ============================= plot options =======================

PLOT = True
PLOT_EVERY = 100
PLOT_AFTER = 0

# ====================== Configuration ======================

D_MAX_PHYSICAL = 1 # physical diameter in m
D_MAX_LATTICE = 15 # lattice points

U_MAX_PHYSICAL = 15 # physical velocity in m/s
U_MAX_LATTICE = 0.3 # lattice points

D_PHYSICAL = 0.5 # physical diameter in m
U_PHYSICAL = 10 # physical velocity in m/s

#15 - 0.3
#7.5 - x
D = (D_MAX_LATTICE * D_PHYSICAL) / D_MAX_PHYSICAL # Cylinder diameter unitless
U0 = (U_MAX_LATTICE * U_PHYSICAL) / U_MAX_PHYSICAL # Inlet velocity unitless


# Simulation parameters
# D = 24                 # Cylinder diameter
# U0 =                  # Inlet velocity
TM = 60000             # Total time steps

# Domain size
NX = int(20 * D)            # Grid points in x direction
NY = int(10 * D)            # Grid points in y direction

# Cylinder position
X_OBJ = NX / 2          # Cylinder x position
Y_OBJ = NY / 2          # Cylinder y position

# IB method parameters
N_MARKER = int(4 * D)       # Number of markers on cylinder
N_ITER_MDF = 3         # Multi-direct forcing iterations
IB_MARGIN = 2          # Margin of the IB region to the cylinder

# Physical parameters
RE = 150               # Reynolds number
UR = 5                 # Reduced velocity
MR = 10                # Mass ratio
DR = 0                 # Damping ratio

# =================== Pre-calculations ==================

# structural parameters
FN = U0 / (UR * D)                                          # Natural frequency
MASS = math.pi * (D / 2) ** 2 * MR                          # Mass of the cylinder
STIFFNESS = (FN * 2 * math.pi) ** 2 * MASS * (1 + 1 / MR)   # Stiffness of the spring
DAMPING = 2 * math.sqrt(STIFFNESS * MASS) * DR              # Damping of the spring

# fluid parameters
NU = U0 * D / RE                                            # Kinematic viscosity
TAU = 3 * NU + 0.5                                          # Relaxation time
OMEGA = 1 / TAU                                             # Relaxation parameter
MRT_COL_LEFT, MRT_SRC_LEFT = mrt.precompute_left_matrices(OMEGA)

# eulerian meshgrid
X, Y = jnp.meshgrid(jnp.arange(NX, dtype=jnp.int32), 
                    jnp.arange(NY, dtype=jnp.int32), 
                    indexing="ij")

# lagrangian markers
THETA_MAKERS = jnp.linspace(0, jnp.pi * 2, N_MARKER, dtype=jnp.float32, endpoint=False)
X_MARKERS = X_OBJ + 0.5 * D * jnp.cos(THETA_MAKERS)
Y_MARKERS = Y_OBJ + 0.5 * D * jnp.sin(THETA_MAKERS)
L_ARC = D * math.pi / N_MARKER

# dynamic ibm region
IB_START_X = int(X_OBJ - 0.5 * D - IB_MARGIN)
IB_START_Y = int(Y_OBJ - 0.5 * D - IB_MARGIN)
IB_SIZE = int(D + IB_MARGIN * 2)

# =================== define variables ==================

# fluid variables
rho = jnp.ones((NX, NY), dtype=jnp.float32)      # density of fluid
u = jnp.zeros((2, NX, NY), dtype=jnp.float32)    # velocity of fluid
f = jnp.zeros((9, NX, NY), dtype=jnp.float32)    # distribution functions
feq = jnp.zeros((9, NX, NY), dtype=jnp.float32)  # equilibrium distribution functions

# structural variables
d = jnp.zeros((2), dtype=jnp.float32)   # displacement of cylinder
v = jnp.zeros((2), dtype=jnp.float32)   # velocity of cylinder
a = jnp.zeros((2), dtype=jnp.float32)   # acceleration of cylinder
h = jnp.zeros((2), dtype=jnp.float32)   # hydrodynamic force

# initial conditions
u = u.at[0].set(U0)
f = lbm.get_equilibrium(rho, u)
v = d.at[1].set(1e-2)  # add an initial velocity to the cylinder
feq_init = f[:,0,0]


# =================== define calculation routine ===================

j = -1
@jax.jit
def update(f, d, v, a, h):
    global j
    j += 1
   
    # update new macroscopic
    rho, u = lbm.get_macroscopic(f)

    
    # Collision
    feq = lbm.get_equilibrium(rho, u)
    f = mrt.collision(f, feq, MRT_COL_LEFT)
      
    # update markers position
    x_markers, y_markers = ib.get_markers_coords_2dof(X_MARKERS, Y_MARKERS, d)
    
    # update ibm region
    ib_start_x = (IB_START_X + d[0]).astype(jnp.int32)
    ib_start_y = (IB_START_Y + d[1]).astype(jnp.int32)
    
    # extract data from ibm region
    u_slice = jax.lax.dynamic_slice(u, (0, ib_start_x, ib_start_y), (2, IB_SIZE, IB_SIZE))
    X_slice = jax.lax.dynamic_slice(X, (ib_start_x, ib_start_y), (IB_SIZE, IB_SIZE))
    Y_slice = jax.lax.dynamic_slice(Y, (ib_start_x, ib_start_y), (IB_SIZE, IB_SIZE))
    f_slice = jax.lax.dynamic_slice(f, (0, ib_start_x, ib_start_y), (9, IB_SIZE, IB_SIZE))
    

    v_markers = jnp.tile(v.reshape((1, 2)), (N_MARKER, 1))
    # calculate ibm force
    g_slice, h_markers = ib.multi_direct_forcing(u_slice, X_slice, Y_slice, 
                                                   v_markers, x_markers, y_markers, N_MARKER, L_ARC, 
                                                   N_ITER_MDF, ib.kernel_range4)
    
    # apply the force to the lattice
    g_lattice = lbm.get_discretized_force(g_slice, u_slice)
    s_slice = mrt.get_source(g_lattice, MRT_SRC_LEFT)    
    # f = dynamic_update_slice(f, f_slice + s_slice, (0, ib_start_x, ib_start_y))
    f = jax.lax.dynamic_update_slice(f, f_slice + s_slice, (0, ib_start_x, ib_start_y))

    # apply the force to the cylinder
    h = ib.get_force_to_obj(h_markers)
    # h += a * math.pi * D ** 2 / 4   
    scale = (math.pi  * D * D) / 4
    h = jnp.add(h, jnp.multiply(a, scale))
    a, v, d = dyn.newmark_2dof(a, v, d, h, MASS, STIFFNESS, DAMPING)

    # Streaming
    f = lbm.streaming(f)

    # Boundary conditions
    f = lbm.boundary_equilibrium(f, feq_init[:,jnp.newaxis], loc='right')
    f = lbm.velocity_boundary(f, U0, 0, loc='left')
     
    return f, rho, u, d, v, a, h


# =============== create plot template ================

if PLOT:
    mpl.rcParams['figure.raise_window'] = False
    
    plt.figure(figsize=(8, 4))
    
    curl = post.calculate_curl(u)
    im = plt.imshow(
        curl.T,
        extent=[0, NX/D, 0, NY/D],
        cmap="seismic",
        aspect="equal",
        origin="lower",
        norm=mpl.colors.CenteredNorm(),
    )

    plt.colorbar()
    plt.title(f"velocity {U0}; diameter: {D}")
    plt.xlabel("x/D")
    plt.ylabel("y/D")

    # draw a circle representing the cylinder
    circle = plt.Circle(((X_OBJ + d[0]) / D, (Y_OBJ + d[1]) / D), 0.5, 
                        edgecolor='black', linewidth=0.5,
                        facecolor='white', fill=True)
    plt.gca().add_artist(circle)
    
    # mark the initial position of the cylinder
    plt.plot((X_OBJ + d[0]) / D, Y_OBJ / D, marker='+', markersize=10, color='k', linestyle='None', markeredgewidth=0.5)
    
    plt.tight_layout()

# =============== start simulation ===============

# for t in tqdm(range(TM)):
#     f, rho, u, d, v, a, h = update(f, d, v, a, h)


#     if PLOT and t % PLOT_EVERY == 0 and t > PLOT_AFTER:
#         im.set_data(post.calculate_curl(u).T)
#         im.autoscale()
#         circle.center = ((X_OBJ + d[0]) / D, (Y_OBJ + d[1]) / D)        
#         plt.pause(0.01)


def updatePublic():
    global f, d, v, a, h
    f, rho, u, d, v, a, h = update(f, d, v, a, h)
    
    return f, rho, u, d, v, a, h









