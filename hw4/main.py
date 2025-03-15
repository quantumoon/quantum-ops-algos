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
    N = w.shape[0]
    n_gates = 0
    for layer in range(1, p + 1):
        if BCH:
            state = apply_comm(state, t / p, gamma, w)
            n_gates += (N - 1) * N
        state = apply_ZZ(state, t / p, gamma)
        state = apply_mixer(state, t / p, w)
        n_gates += N + (N * (N - 1)) // 2
    return state.reshape(1<<N), n_gates


def SuzukiTrotter_2(state, t, p, gamma, w):
    N = w.shape[0]
    n_gates = 0
    for layer in range(1, p + 1):
        state = apply_mixer(state, 0.5 * t / p, w)
        state = apply_ZZ(state, t / p, gamma)
        state = apply_mixer(state, 0.5 * t / p, w)
        n_gates += 2 * N + (N * (N - 1)) // 2
    return state.reshape(1<<N), n_gates


def SuzukiTrotter_4(state, t, p, gamma, w):
    N = w.shape[0]
    b = (4 - 4**(1/3))**-1
    n_gates = 0
    for layer in range(1, p + 1):
        state = apply_mixer(state, b*t/(2*p), w)
        state = apply_ZZ(state, b*t/p, gamma)
        state = apply_mixer(state, b*t/p, w)
        state = apply_ZZ(state, b*t/p, gamma)
        n_gates += 2 * N + N * (N - 1)
        
        state = apply_mixer(state, (1-3*b)*t/(2*p), w)
        state = apply_ZZ(state, (1-4*b)*t/p, gamma)
        state = apply_mixer(state, (1-3*b)*t/(2*p), w)
        n_gates += 2 * N + (N * (N - 1)) // 2

        state = apply_ZZ(state, b*t/p, gamma)
        state = apply_mixer(state, b*t/p, w)
        state = apply_ZZ(state, b*t/p, gamma)
        state = apply_mixer(state, b*t/(2*p), w)
        n_gates += 2 * N + N * (N - 1)
        
    return state.reshape(1<<N), n_gates


np.random.seed(42)

N = 6
max_p = 30
t = 2.

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
    
plt.figure(dpi=200)
labels = ['1st order of ST', '1st order of ST + BCH', '2nd order of ST', '4th order of ST']
for i, fidelity in enumerate(fidelities):
    plt.plot(Xs[i], fidelity, label=labels[i], marker='.', markersize=6)
plt.xlabel('Number of gates')
plt.ylabel('Fidelity')
plt.title('Fidelity between true and approximate solutions')
plt.legend()
plt.grid(True)
plt.xscale('log')
plt.xlim(left=1)
plt.savefig('fidelity_results.png')