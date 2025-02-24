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
fidelity_no_echo = []
purity_no_echo = []
fidelity_echo = []
purity_echo = []
for t in np.arange(0, tau, dt):
    system_rho = np.sum(total_rho, axis=-1)  # partial trace
    time.append(t)
    fidelity_no_echo.append(np.trace(system_rho @ system_rho_init).real)
    purity_no_echo.append(np.trace(system_rho @ system_rho).real)
    fidelity_echo.append(fidelity_no_echo[-1])
    purity_echo.append(purity_no_echo[-1])

    total_rho = U[:, None, :] * total_rho * U[None, :, :]  # shape: (2, 2, 2^N)

# Apply R_X(π) to the first qubit (system qubit)
# Note: It is easy to show that a π pulse is equivalent to swapping the blocks
# along the main and side diagonals of the density matrix
total_rho_echo = np.copy(total_rho)
total_rho_echo[0,0], total_rho_echo[1,1] = total_rho_echo[1,1], total_rho_echo[0,0]
total_rho_echo[0,1], total_rho_echo[1,0] = total_rho_echo[1,0], total_rho_echo[0,1]

for t in np.arange(tau, 2 * tau + dt, dt):
    system_rho = np.sum(total_rho, axis=-1)  # partial trace
    system_rho_echo = np.sum(total_rho_echo, axis=-1)
    time.append(t)
    fidelity_no_echo.append(np.trace(system_rho @ system_rho_init).real)
    purity_no_echo.append(np.trace(system_rho @ system_rho).real)
    fidelity_echo.append(np.trace(system_rho_echo @ system_rho_init).real)
    purity_echo.append(np.trace(system_rho_echo @ system_rho_echo).real)

    total_rho = U[:, None, :] * total_rho * U[None, :, :]  # shape: (2, 2, 2^N)
    total_rho_echo = U[:, None, :] * total_rho_echo * U[None, :, :]

# 4. Rendering
fig, axs = plt.subplots(2, 1, figsize=(8, 6), dpi=200)

axs[0].plot(time, purity_no_echo, color='blue', linestyle='--', linewidth=2, label='No echo')
axs[0].plot(time, purity_echo, color='blue', linewidth=2, label='Echo')
axs[0].axvline(x=tau, color='red', linestyle='--', linewidth=2, label=r"Echo's $\pi$ pulse")
axs[0].set_title('Purity of the quantum system', fontsize=14)
axs[0].set_xlabel('Time', fontsize=12)
axs[0].set_ylabel('Purity', fontsize=12)
axs[0].legend(fontsize=8)

axs[1].plot(time, fidelity_no_echo, color='green', linestyle='--', linewidth=2, label='No echo')
axs[1].plot(time, fidelity_echo, color='green', linewidth=2, label='Echo')
axs[1].axvline(x=tau, color='red', linestyle='--', linewidth=2, label=r"Echo's $\pi$ pulse")
axs[1].set_title('Fidelity between current and initial quantum states', fontsize=14)
axs[1].set_xlabel('Time', fontsize=12)
axs[1].set_ylabel('Fidelity', fontsize=12)
axs[1].legend(fontsize=8)

plt.tight_layout()
plt.show()