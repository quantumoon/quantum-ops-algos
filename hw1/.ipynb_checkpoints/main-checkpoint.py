import numpy as np
from solvers import *
from animation_utils import animate_wavefunction

# 1. Solving equation with inplicit scheme & 3-diag matrix 
N = 10000  # x steps
K = 400  # time steps
T = 10.
L = 100.
time = np.linspace(0, T, K + 1)
x = np.linspace(0, L, N + 1)

## 1.1. Barrier with V_0 > 0
loc = 30.
scale = 1.
V_0 = 10.
x_V = 50.
h = 1.
k = 7.

solver = BandedSolver(L=L, T=T, N=N, K=K, V_0=V_0, h=h, x_V=x_V, loc=loc, scale=scale, k=k)
solver.run()
psi_matrix = solver.history
animate_wavefunction(psi_matrix, x, time, V_0, x_V, h, x_lims=(20, 75), filename=r"gifs\BandedSolver_barrier.gif")

## 1.2 No barrier
loc = 90.
scale = 0.5
V_0 = 0.
x_V = 0.
h = 0.
k = -5.

solver = BandedSolver(L=L, T=T, N=N, K=K, V_0=V_0, h=h, x_V=x_V, loc=loc, scale=scale, k=k)
solver.run()
psi_matrix = solver.history
animate_wavefunction(psi_matrix, x, time, V_0, x_V, h, x_lims=(50, 100), filename=r"gifs\BandedSolver_no_barrier.gif")


# 2. Solving equation with explicit state vector evolution
loc = 40.
scale = 1.5
V_0 = 5.
x_V = 50.
h = 0.2
k = 2.

solver = EvolutionSolver(L=L, T=T, N=N, K=K, V_0=V_0, h=h, x_V=x_V, loc=loc, scale=scale, recompute_H=False, h_shape=100, k=k)
solver.run()
psi_matrix = solver.history
animate_wavefunction(psi_matrix, x, time, V_0, x_V, h, x_lims=(30, 70), filename=r"gifs\EvolutionSolver_barrier.gif")