import numpy as np
import matplotlib.pyplot as plt
from numpy import random

random.seed(42)

N = 16  # environment qubits
kT = 1.
dt = 0.01
tau = 1.
gamma_bound = 0.5
# Since the Hamiltonian and environment's initial density matrix are diag-like,
# We will keep only the main diagonals for all matrices
Pauli_Z = np.array([1., -1.])

def Z(n):
    I_Nn = np.ones(1 << (N - n))
    return np.kron(np.ones(1 << (n - 1)),
                   np.hstack((I_Nn, -I_Nn)))

# 1. Interaction Hamiltonian & evolution operator initialization
gammas = random.uniform(low=-gamma_bound, high=gamma_bound, size=N)
H = np.einsum('i,ij', gammas, np.array([np.kron(Pauli_Z, Z(n)) for n in range(1, N + 1)]))  # shape: (2^(N+1), )
U = np.exp(-1j * H * dt).reshape(2, -1)   # shape: (2, 2^N)

# 2. System & environment initialization
# |phi><phi|, here |phi> = |+> - system's initial state
system_rho_init = np.full((2, 2), 0.5, dtype=np.complex128)

energies = random.exponential(scale=1.0, size=(1 << N))
env_rho_init = np.exp(-energies / kT)
env_rho_init /= np.sum(env_rho_init) # shape: (2^N, )

total_rho = np.tensordot(system_rho_init, env_rho_init, axes=0)  # shape: (2, 2, 2^N)

# 3. Evolution simulation
time = []
fidelity = []
purity = []
for t in np.arange(0, tau, dt):
    system_rho = np.sum(total_rho, axis=-1)  # partial trace
    time.append(t)
    fidelity.append(np.trace(system_rho @ system_rho_init).real)
    purity.append(np.trace(system_rho @ system_rho).real)

    total_rho = U[:, None, :] * total_rho * U[None, :, :]  # shape: (2, 2, 2^N)

# Apply R_X(π) to each of the N+1 qubits
# Note: It is easy to show that a π pulse is equivalent to reversing the order of elements
# along the last axis of the density matrix
total_rho = total_rho[..., ::-1]

for t in np.arange(tau, 2 * tau + dt, dt):
    system_rho = np.sum(total_rho, axis=-1)  # partial trace
    time.append(t)
    fidelity.append(np.trace(system_rho @ system_rho_init).real)
    purity.append(np.trace(system_rho @ system_rho).real)

    total_rho = U[:, None, :] * total_rho * U[None, :, :]  # shape: (2, 2, 2^N)

# 4. Rendering
fig, axs = plt.subplots(2, 1, figsize=(8, 6), dpi=200)

axs[0].plot(time, purity, color='blue', linewidth=2)
axs[0].axvline(x=tau, color='red', linestyle='--', linewidth=2, label=r"Echo's $\pi$ pulse")
axs[0].set_title('Purity of the quantum system', fontsize=14)
axs[0].set_xlabel('Time', fontsize=12)
axs[0].set_ylabel('Purity', fontsize=12)
axs[0].legend(fontsize=12)

axs[1].plot(time, fidelity, color='green', linewidth=2)
axs[1].axvline(x=tau, color='red', linestyle='--', linewidth=2, label=r"Echo's $\pi$ pulse")
axs[1].set_title('Fidelity between current and initial quantum states', fontsize=14)
axs[1].set_xlabel('Time', fontsize=12)
axs[1].set_ylabel('Fidelity', fontsize=12)
axs[1].legend(fontsize=12)

plt.tight_layout()
plt.show()