import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from simulator_utils import *


def generate_hamiltonian(N, w, gamma):    
    sigma_x = np.array([[0, 1],
                        [1, 0]], dtype=np.complex64)
    sigma_z = np.array([[1, 0],
                        [0, -1]], dtype=np.complex64)
    I = np.eye(2, dtype=np.complex64)
    
    A_terms = np.array([w[i] * reduce(np.kron, [I] * i + [sigma_x] + [I] * (N - i - 1)) for i in range(N)])
    A = np.sum(A_terms, axis=0)
    
    B_terms = np.array([gamma[i, j] * reduce(np.kron, [I] * i + [sigma_z] + [I] * (j - i - 1) + [sigma_z] + [I] * (N - j - 1))
                        for i in range(N) for j in range(i + 1, N)])
    B = np.sum(B_terms, axis=0)
    
    return A + B


def fidelity(psi_true, psi_pred):
    return np.abs(np.dot(psi_true, psi_pred.conj()))**2


def SuzukiTrotter_1(state, t, p, gamma, w, BCH=False):
    n_gates = 0
    for layer in range(1, p + 1):
        if BCH:
            state, c = apply_comm(state, t / p, gamma, w)
            n_gates += c
        state, c = apply_ZZ(state, t / p, gamma)
        n_gates += c
        state, c = apply_mixer(state, t / p, w)
        n_gates += c
    return state.reshape(1<<N), n_gates


def SuzukiTrotter_2(state, t, p, gamma, w):
    n_gates = 0
    for layer in range(1, p + 1):
        state, c = apply_mixer(state, 0.5 * t / p, w)
        n_gates += c
        state, c = apply_ZZ(state, t / p, gamma)
        n_gates += c
        state, c = apply_mixer(state, 0.5 * t / p, w)
        n_gates += c
    return state.reshape(1<<N), n_gates


def SuzukiTrotter_4(state, t, p, gamma, w):
    b = (4 - 4**(1/3))**-1
    n_gates = 0
    for layer in range(1, p + 1):
        state, c = apply_mixer(state, b*t/(2*p), w)
        n_gates += c
        state, c = apply_ZZ(state, b*t/p, gamma)
        n_gates += c
        state, c = apply_mixer(state, b*t/p, w)
        n_gates += c
        state, c = apply_ZZ(state, b*t/p, gamma)
        n_gates += c
        
        state, c = apply_mixer(state, (1-3*b)*t/(2*p), w)
        n_gates += c
        state, c = apply_ZZ(state, (1-4*b)*t/p, gamma)
        n_gates += c
        state, c = apply_mixer(state, (1-3*b)*t/(2*p), w)
        n_gates += c

        state, c = apply_ZZ(state, b*t/p, gamma)
        n_gates += c
        state, c = apply_mixer(state, b*t/p, w)
        n_gates += c
        state, c = apply_ZZ(state, b*t/p, gamma)
        n_gates += c
        state, c = apply_mixer(state, b*t/(2*p), w)
        n_gates += c
        
    return state.reshape(1<<N), n_gates


max_p = 20
labels = ['1st order of ST', '1st order of ST + BCH', '2nd order of ST', '4th order of ST']

times = [1, 5, 10]
fixed_N = 6

qubits = [2, 5, 10]
fixed_t = 2

fig, axes = plt.subplots(2, 3, dpi=200, figsize=(18, 10))

for i, t in enumerate(times):
    N = fixed_N
    np.random.seed(42)
    w = np.random.uniform(0, 2, size=N)
    gamma = np.random.uniform(0, 2, size=(N, N))
    
    H = generate_hamiltonian(N, w, gamma)
    psi_init = np.random.randn(1<<N) + 1j * np.random.randn(1<<N)
    psi_init /= np.sqrt(np.sum(np.abs(psi_init)**2))
    
    psi_true = expm(-1j * H * t) @ psi_init
    psi_init = psi_init.reshape([2] * N)
    
    fidelities = np.zeros((4, max_p + 1))
    Xs = np.zeros((4, max_p + 1))
    for p in range(1, max_p + 1):
        psi_ST1, n1 = SuzukiTrotter_1(psi_init.copy(), t, p, gamma, w, BCH=False)
        psi_ST1_BCH, n1_BCH = SuzukiTrotter_1(psi_init.copy(), t, p, gamma, w, BCH=True)
        psi_ST2, n2 = SuzukiTrotter_2(psi_init.copy(), t, p, gamma, w)
        psi_ST4, n4 = SuzukiTrotter_4(psi_init.copy(), t, p, gamma, w)
        
        Xs[:, p] = np.array([n1, n1_BCH, n2, n4])
        fidelities[:, p] = np.array([fidelity(psi_true, psi_ST1),
                                      fidelity(psi_true, psi_ST1_BCH),
                                      fidelity(psi_true, psi_ST2),
                                      fidelity(psi_true, psi_ST4)])
    
    ax = axes[0, i]
    for j in range(4):
        ax.plot(Xs[j], fidelities[j], label=labels[j], marker='.', markersize=6)
    ax.set_xlabel('Number of gates')
    ax.set_ylabel('Fidelity')
    ax.set_title(f'N = {N}, t = {t}')
    ax.legend()
    ax.grid(True)
    ax.set_xscale('log')
    ax.set_xlim(left=1)

for i, N in enumerate(qubits):
    np.random.seed(42)
    w = np.random.uniform(0, 2, size=N)
    gamma = np.random.uniform(0, 2, size=(N, N))
    
    H = generate_hamiltonian(N, w, gamma)
    psi_init = np.random.randn(1<<N) + 1j * np.random.randn(1<<N)
    psi_init /= np.sqrt(np.sum(np.abs(psi_init)**2))
    
    psi_true = expm(-1j * H * fixed_t) @ psi_init
    psi_init = psi_init.reshape([2] * N)
    
    fidelities = np.zeros((4, max_p + 1))
    Xs = np.zeros((4, max_p + 1))
    for p in range(1, max_p + 1):
        psi_ST1, n1 = SuzukiTrotter_1(psi_init.copy(), fixed_t, p, gamma, w, BCH=False)
        psi_ST1_BCH, n1_BCH = SuzukiTrotter_1(psi_init.copy(), fixed_t, p, gamma, w, BCH=True)
        psi_ST2, n2 = SuzukiTrotter_2(psi_init.copy(), fixed_t, p, gamma, w)
        psi_ST4, n4 = SuzukiTrotter_4(psi_init.copy(), fixed_t, p, gamma, w)
        
        Xs[:, p] = np.array([n1, n1_BCH, n2, n4])
        fidelities[:, p] = np.array([fidelity(psi_true, psi_ST1),
                                      fidelity(psi_true, psi_ST1_BCH),
                                      fidelity(psi_true, psi_ST2),
                                      fidelity(psi_true, psi_ST4)])
    
    ax = axes[1, i]
    for j in range(4):
        ax.plot(Xs[j], fidelities[j], label=labels[j], marker='.', markersize=6)
    ax.set_xlabel('Number of gates')
    ax.set_ylabel('Fidelity')
    ax.set_title(f'N = {N}, t = {fixed_t}')
    ax.legend()
    ax.grid(True)
    ax.set_xscale('log')
    ax.set_xlim(left=1)

plt.tight_layout()
plt.savefig('fidelity_results_subplots.png')