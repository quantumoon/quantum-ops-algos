import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax.scipy.linalg import expm
from matrix_utils import *

N = 5
max_p = 200
t = 10.

A, B = map(jnp.array, generate_hamiltonian(N))

np.random.seed(42)
psi_init = np.random.randn(1<<N) + 1j * np.random.randn(1<<N)
psi_init /= np.sqrt(np.sum(np.abs(psi_init)**2))

psi_true = expm(-1j * (A + B) * t) @ psi_init

fidelities = np.zeros((4, max_p + 1))
for p in range(1, max_p + 1):
    ops = map(np.array, (S_1(-1j*t / p, A, B),
                         S_1_5(-1j*t / p, A, B),
                         S_2(-1j*t / p, A, B),
                         S_4(-1j*t / p, A, B)))
    for i, u in enumerate(ops):
        psi_pred = u @ psi_init
        for _ in range(p - 1):
            psi_pred = u @ psi_pred
        fidelities[i, p] = fidelity(psi_true, psi_pred)

plt.figure(dpi=200)
X = np.arange(max_p + 1)
labels = ['1st order of ST', '1st order of ST + BCH', '2nd order of ST', '4th order of ST']
for i, fidelity in enumerate(fidelities):
    plt.plot(X, fidelity, label=labels[i])
plt.xlabel('Trotterization steps')
plt.xlim(0, max_p)
plt.ylabel('Fidelity')
plt.title('Fidelity between true and approximate solutions')
plt.legend()
plt.savefig('fidelity_results.png')