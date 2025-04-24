import numpy as np


def rx(theta: float):
    return np.array([[np.cos(theta / 2), -1j * np.sin(theta / 2)],
                     [-1j * np.sin(theta / 2), np.cos(theta / 2)]])


def rzz(theta: float):
    return np.diag(np.exp(-1j * theta * np.array([1,-1,-1,1]) / 2)).reshape(2,2,2,2)


def zz_hamiltonian_eigens(N: int, compute_honestly=True):
    if not compute_honestly:
        bitstrings = np.array([1 if i % 2 else 0 for i in range(N)])
        bitstrings = np.array([bitstrings, 1 - bitstrings])
        ids = np.array([int(''.join(map(str, bitstring)), base=2) for bitstring in bitstrings])
        return bitstrings, ids
    dim = 1 << N
    energies = np.zeros(dim, dtype=int)
    for idx in range(dim):
        e = 0
        for i in range(N - 1):
            b1 = (idx >> i) & 1
            b2 = (idx >> (i + 1)) & 1
            z1 = 1 - 2 * b1
            z2 = 1 - 2 * b2
            e += z1 * z2
        energies[idx] = e
    ground_energy = energies.min()
    ground_states = np.where(energies == ground_energy)[0]
    bitstrings = np.array([list(map(int, np.binary_repr(state, width=N))) for state in ground_states])
    return bitstrings, ground_states