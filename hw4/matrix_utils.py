import numpy as np
import jax.numpy as jnp
from jax.scipy.linalg import expm
from jax import jit
from functools import reduce


def generate_hamiltonian(N, seed=42):
    np.random.seed(seed)
    
    w = np.random.uniform(0, 2, size=N)
    gamma = np.random.uniform(0, 2, size=(N, N))
    
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
    
    return A, B


def fidelity(psi_true, psi_pred):
    return np.abs(np.dot(psi_true, psi_pred.conj()))**2


@jit
def S_1(t, A, B):
    return expm(A * t) @ expm(B * t)


@jit
def S_1_5(t, A, B):
    return S_1(t, A, B) @ expm(-(A @ B - B @ A) * t**2 / 2)


@jit
def S_2(t, A, B):
    return expm(A * t / 2) @ expm(B * t) @ expm(A * t / 2)


@jit
def S_4(t, A, B):
    betta = (4 - 4**(1/3))**-1
    return S_2(betta * t, A, B) @ S_2(betta * t, A, B) @ S_2((1. - 4. * betta) * t, A, B) @ S_2(betta * t, A, B) @ S_2(betta * t, A, B)