import numpy as np


def random_zz_max_energy(N: int, proba: float, seed: int = 42):
    np.random.seed(seed)
    
    pairs = [(i, j) for i in range(N) for j in range(i) if np.random.rand() < proba]
    dim = 1 << N
    energies = np.zeros(dim, dtype=int)
    for idx in range(dim):
        z = np.array([1 - 2 * ((idx >> k) & 1) for k in range(N)], dtype=int)
        energies[idx] = sum(z[i] * z[j] for (i, j) in pairs)

    max_energy = energies.max()
    state_ids = np.where(energies == max_energy)[0]

    bitstrings = np.array([list(map(int, np.binary_repr(s, width=N))) for s in state_ids], dtype=int)

    return bitstrings, state_ids, pairs


def rx(theta: float) -> np.ndarray:
    return np.array([[np.cos(theta / 2), -1j * np.sin(theta / 2)],
                     [-1j * np.sin(theta / 2), np.cos(theta / 2)]])


def rzz(theta: float) -> np.ndarray:
    return np.diag(np.exp(-1j * theta * np.array([1,-1,-1,1]) / 2)).reshape(2,2,2,2)