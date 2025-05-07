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


class VortexSimulation:
    def __init__(self, D, U0, RE, UR, MR, DR, NX, NY, PLOT=False):
        # ===== Plot Settings =====
        self.PLOT = PLOT
        self.PLOT_EVERY = 100
        self.PLOT_AFTER = 0

        # ===== Configuration =====

        self.D = D
        self.U0 = U0

        self.RE = RE
        self.UR = UR
        self.MR = MR
        self.DR = DR

        self.NX = NX
        self.NY = NY

        self.X_OBJ = self.NX / 2
        self.Y_OBJ = self.NY / 2

        self.N_MARKER = int(4 * self.D)
        self.N_ITER_MDF = 3
        self.IB_MARGIN = 2

        self.FN = self.U0 / (self.UR * self.D)
        self.MASS = math.pi * (self.D / 2) ** 2 * self.MR
        self.STIFFNESS = (self.FN * 2 * math.pi) ** 2 * self.MASS * (1 + 1 / self.MR)
        self.DAMPING = 2 * math.sqrt(self.STIFFNESS * self.MASS) * self.DR

        self.NU = self.U0 * self.D / self.RE
        self.TAU = 3 * self.NU + 0.5
        self.OMEGA = 1 / self.TAU
        self.MRT_COL_LEFT, self.MRT_SRC_LEFT = mrt.precompute_left_matrices(self.OMEGA)

        self.X, self.Y = jnp.meshgrid(jnp.arange(self.NX), jnp.arange(self.NY), indexing="ij")

        self.THETA_MARKERS = jnp.linspace(0, jnp.pi * 2, self.N_MARKER, endpoint=False, dtype=jnp.float32)
        self.X_MARKERS = self.X_OBJ + 0.5 * self.D * jnp.cos(self.THETA_MARKERS)
        self.Y_MARKERS = self.Y_OBJ + 0.5 * self.D * jnp.sin(self.THETA_MARKERS)
        self.L_ARC = self.D * math.pi / self.N_MARKER

        self.IB_START_X = int(self.X_OBJ - 0.5 * self.D - self.IB_MARGIN)
        self.IB_START_Y = int(self.Y_OBJ - 0.5 * self.D - self.IB_MARGIN)
        self.IB_SIZE = int(self.D + self.IB_MARGIN * 2)

        # ====== Variables ======
        self.rho = jnp.ones((self.NX, self.NY), dtype=jnp.float32)
        self.u = jnp.zeros((2, self.NX, self.NY), dtype=jnp.float32)
        self.f = jnp.zeros((9, self.NX, self.NY), dtype=jnp.float32)
        self.feq = jnp.zeros((9, self.NX, self.NY), dtype=jnp.float32)

        self.d = jnp.zeros((2,), dtype=jnp.float32)
        self.v = jnp.zeros((2,), dtype=jnp.float32).at[1].set(1e-2)
        self.a = jnp.zeros((2,), dtype=jnp.float32)
        self.h = jnp.zeros((2,), dtype=jnp.float32)

        self.u = self.u.at[0].set(self.U0)
        self.f = lbm.get_equilibrium(self.rho, self.u)
        self.feq_init = self.f[:, 0, 0]

        if self.PLOT:
            self.setup_plot()


    @staticmethod
    def update(
        f: jnp.ndarray,
        d: jnp.ndarray,
        v: jnp.ndarray,
        a: jnp.ndarray,
        h: jnp.ndarray,
        X_MARKERS: jnp.ndarray,
        Y_MARKERS: jnp.ndarray,
        N_MARKER: int,
        L_ARC: float,
        MRT_COL_LEFT: jnp.ndarray,
        MRT_SRC_LEFT: jnp.ndarray,
        N_ITER_MDF: int,
        IB_START_X: int,
        IB_START_Y: int,
        IB_SIZE: int,
        MASS: float,
        STIFFNESS: float,
        DAMPING: float,
        feq_init: jnp.ndarray,
        U0: float,
        D: float,
        X: jnp.ndarray,
        Y: jnp.ndarray,
    ):
        """
        JIT-compiled update function for the VortexSimulation class.
        """
        # Update macroscopic variables
        rho, u = lbm.get_macroscopic(f)

        # Collision step
        feq = lbm.get_equilibrium(rho, u)
        f = mrt.collision(f, feq, MRT_COL_LEFT)

        # Marker coordinates update
        x_markers, y_markers = ib.get_markers_coords_2dof(X_MARKERS, Y_MARKERS, d)

        # Determine slicing start positions
        ib_start_x = (IB_START_X + d[0]).astype(jnp.int32)
        ib_start_y = (IB_START_Y + d[1]).astype(jnp.int32)

        # Extract region of interest using IB_SIZE
        u_slice = jax.lax.dynamic_slice(u, (0, ib_start_x, ib_start_y), (2, IB_SIZE, IB_SIZE))
        X_slice = jax.lax.dynamic_slice(X, (ib_start_x, ib_start_y), (IB_SIZE, IB_SIZE))
        Y_slice = jax.lax.dynamic_slice(Y, (ib_start_x, ib_start_y), (IB_SIZE, IB_SIZE))
        f_slice = jax.lax.dynamic_slice(f, (0, ib_start_x, ib_start_y), (9, IB_SIZE, IB_SIZE))
        
        # Apply forcing
        v_markers = jnp.tile(v.reshape((1, 2)), (N_MARKER, 1))
        g_slice, h_markers = ib.multi_direct_forcing(
            u_slice, X_slice, Y_slice, v_markers, x_markers, y_markers, N_MARKER, L_ARC, N_ITER_MDF, ib.kernel_range4
        )

        # Update force
        g_lattice = lbm.get_discretized_force(g_slice, u_slice)
        s_slice = mrt.get_source(g_lattice, MRT_SRC_LEFT)
        f = jax.lax.dynamic_update_slice(f, f_slice + s_slice, (0, ib_start_x, ib_start_y))

        # Apply force to the cylinder
        h = ib.get_force_to_obj(h_markers)
        scale = (math.pi * D ** 2) / 4
        h = jnp.add(h, jnp.multiply(a, scale))
        a, v, d = dyn.newmark_2dof(a, v, d, h, MASS, STIFFNESS, DAMPING)

        # Streaming and boundary conditions
        f = lbm.streaming(f)
        f = lbm.boundary_equilibrium(f, feq_init[:, jnp.newaxis], loc='right')
        f = lbm.velocity_boundary(f, U0, 0, loc='left')

        return f, rho, u, d, v, a, h



    def setup_plot(self):
        mpl.rcParams['figure.raise_window'] = False
        
        plt.figure(figsize=(8, 4))
        
        curl = post.calculate_curl(self.u)
        im = plt.imshow(
            curl.T,
            extent=[0, self.NX / self.D, 0, self.NY / self.D],
            cmap="seismic",
            aspect="equal",
            origin="lower",
            norm=mpl.colors.CenteredNorm(),
        )

        plt.colorbar()
        plt.title(f"velocity {self.U0}; diameter: {self.D}")
        plt.xlabel("x/D")
        plt.ylabel("y/D")

        # draw a circle representing the cylinder
        circle = plt.Circle(((self.X_OBJ + self.d[0]) / self.D, (self.Y_OBJ + self.d[1]) / self.D), 0.5, 
                            edgecolor='black', linewidth=0.5,
                            facecolor='white', fill=True)
        plt.gca().add_artist(circle)
        
        # mark the initial position of the cylinder
        plt.plot((self.X_OBJ + self.d[0]) / self.D, self.Y_OBJ / self.D, marker='+', markersize=10, color='k', linestyle='None', markeredgewidth=0.5)
        
        plt.tight_layout()

# =============== start simulation ===============

    def step(self):

        jitted_update = jax.jit(
            VortexSimulation.update,
            static_argnums=(7, 8, 11, 12, 13, 14, 15, 16, 17)
        )

        (self.f, self.rho, self.u, self.d, self.v,
        self.a, self.h) = jitted_update(
            # 0–4 dynamic core state
            self.f, self.d, self.v, self.a, self.h,
            # 5–6 dynamic marker arrays
            self.X_MARKERS, self.Y_MARKERS,
            # 7–8 static
            self.N_MARKER, self.L_ARC,
            # 9–10 dynamic MRT arrays
            self.MRT_COL_LEFT, self.MRT_SRC_LEFT,
            # 11 static
            self.N_ITER_MDF,
            # 12–14 static grid ints
            self.IB_START_X, self.IB_START_Y, self.IB_SIZE,
            # 15–17 static floats
            self.MASS, self.STIFFNESS, self.DAMPING,
            # 18–22 dynamic arrays/floats
            self.feq_init, self.U0, self.D, self.X, self.Y
        )

        # if self.PLOT and hasattr(self, "im"):
        #     self.im.set_data(post.calculate_curl(self.u).T)
        #     self.im.autoscale()
        #     self.circle.center = ((self.X_OBJ + self.d[0]) / self.D, self.Y_OBJ / self.D)
        #     plt.pause(0.01)
        return self.f, self.rho, self.u, self.d, self.v, self.a, self.h