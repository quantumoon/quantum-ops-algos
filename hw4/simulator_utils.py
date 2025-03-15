import numpy as np
from scipy.linalg import expm


def apply_1qubit_gate(state, U, i):
    result = np.tensordot(U, state, axes=([1], [i]))
    result = np.moveaxis(result, 0, i)
    return result


def apply_2qubit_gate(state, U, i, j):
    if U.shape == (4, 4):
        U = U.reshape(2, 2, 2, 2)
    result = np.tensordot(U, state, axes=([2, 3], [i, j]))
    result = np.moveaxis(result, [0, 1], [i, j])
    return result


def apply_mixer(state, t, w):
    N = w.shape[0]
    for n in range(N):
        U = np.array([[np.cos(w[n] * t), -1j * np.sin(w[n] * t)],
                      [-1j * np.sin(w[n] * t), np.cos(w[n] * t)]], dtype=np.complex64)
        state = apply_1qubit_gate(state, U, n)
    return state


def apply_ZZ(state, t, gamma):
    N = gamma.shape[0]
    for n in range(N):
        for k in range(n + 1, N):
            U = expm(-1j * t * gamma[n, k] * np.diag([1, -1, -1, 1]))
            state = apply_2qubit_gate(state, U, n, k)
    return state


def apply_comm(state, t, gamma, w):
    N = w.shape[0]
    ZY = 1j * np.array([[0, -1, 0, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, 1],
                        [0, 0, -1, 0]], dtype=np.complex64)
    YZ = 1j * np.array([[0, 0, -1, 0],
                        [0, 0, 0, 1],
                        [1, 0, 0, 0],
                        [0, -1, 0, 0]], dtype=np.complex64)
    for n in range(N):
        for k in range(n + 1,  N):
            U_ZY = expm(-1j * t**2 * gamma[k, n] * w[k] * ZY)
            U_YZ = expm(-1j * t**2 * gamma[k, n] * w[n] * YZ)
            state = apply_2qubit_gate(state, U_ZY, n, k)
            state = apply_2qubit_gate(state, U_YZ, n, k)
    return state