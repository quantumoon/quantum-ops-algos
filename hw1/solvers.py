import numpy as np
import pickle
import os
from scipy.linalg import solve_banded, expm
from scipy.stats import norm
from scipy.integrate import quad

class BandedSolver:
    """
    A quantum evolution solver using a banded matrix approach.
    
    This solver evolves a quantum wavefunction in time using a discretized version of the 
    Schrödinger equation. The spatial discretization is performed on a grid of N+1 points in 
    the interval [0, L], and the evolution is computed using a banded matrix (tridiagonal form)
    that represents the finite-difference approximation of the kinetic energy operator along with 
    a position-dependent potential.
    """
    
    def __init__(self,
                 L: float,
                 T: float,
                 N: int,
                 K: int,
                 V_0: float,
                 x_V: float,
                 h: float,
                 loc: float = 40.,
                 scale: float = 1.,
                 k: float = 1.):

        self.L = L
        self.T = T
        self.N = N
        self.K = K
        self.dx = L / N
        self.dt = T / K
        self.history = []
        
        m = 1  # Particle mass
        gamma = 1j * self.dt / (2 * m * self.dx**2)
        V = lambda x: V_0 if x_V <= x <= x_V + h else 0
        X = np.linspace(0, L, N + 1)

        # Initialize the wavefunction using a Gaussian profile with a phase factor.
        self.psi = np.sqrt(norm.pdf(X, loc=loc, scale=scale)) * np.exp(1j * k * X)
        norm_c = np.sqrt(np.sum(np.abs(self.psi)**2) * self.dx)
        self.psi /= norm_c

        # Initialize the banded matrix (tridiagonal).
        # The banded_matrix is a (3, N+1) array where:
        #   - The middle row (index 1) holds the diagonal.
        #   - The top row (index 0) holds the superdiagonal.
        #   - The bottom row (index 2) holds the subdiagonal.
        self.banded_matrix = np.zeros((3, N + 1), dtype=np.complex128)
        self.banded_matrix[1, 0], self.banded_matrix[1, -1] = 1., 1.
        self.banded_matrix[0, 2:] += -gamma * np.ones(N - 1)
        self.banded_matrix[-1, :-2] += -gamma * np.ones(N - 1)
        self.banded_matrix[1, 1:-1] += np.array([1 + 1j * self.dt * V(x) + 2 * gamma for x in X[1:-1]])

    def run(self):
        """
        Evolve the wavefunction over time using the banded matrix solver.
        """
        for _ in range(self.K + 1):
            self.history.append(self.psi)
            self.psi = solve_banded(l_and_u=(1, 1), ab=self.banded_matrix, b=self.psi)
            norm_c = np.sqrt(np.sum(np.abs(self.psi)**2) * self.dx)
            self.psi /= norm_c


class EvolutionSolver:
    """
    A quantum evolution solver using basis expansion and Hamiltonian exponentiation.
    
    This solver projects an initial quantum state onto a basis (sine functions), computes the 
    Hamiltonian matrix using numerical integration for the potential energy part, and then obtains 
    the time evolution operator via matrix exponentiation. The state is evolved by repeatedly 
    applying this operator to the coefficient vector in the basis representation.
    """
    
    def __init__(self,
                 L: float,
                 T: float,
                 N: int,
                 K: int,
                 V_0: float,
                 x_V: float,
                 h: float,
                 loc: float = 40.,
                 scale: float = 1.,
                 k: float = 1.,
                 recompute_H: bool = False,
                 h_shape: int = 100):

        self.L = L
        self.T = T
        self.N = N
        self.K = K
        self.dt = T / K
        self.dx = L / N
        self.history = []
        self.V_0 = V_0
        self.h = h
        self.x_V = x_V
        self.h_shape = h_shape
        self.X = np.linspace(0, L, N + 1)
        
        n_ = np.arange(1, self.h_shape + 1).reshape(-1, 1)
        self.phi_vals = np.sqrt(2 / self.L) * np.sin(np.pi * n_ * self.X / self.L)  # shape: (h_shape, len(X))
        
        self.m = 1  # Particle mass

        # Define the initial state in spatial representation.
        psi_init = lambda x: np.sqrt(norm.pdf(x, loc=loc, scale=scale)) * np.exp(1j * k * x)
        self.c = np.array([quad(func=lambda x: self.phi(n)(x) * psi_init(x), a=0, b=L, complex_func=True)[0]
                           for n in range(1, h_shape + 1)])

        if recompute_H or not os.path.exists('exp_hamiltonian.pkl'):
            print('COMPUTING HAMILTONIAN...')
            H = self._compute_hamiltonian()
            self.U = expm(-1j * H * self.dt)
            with open('exp_hamiltonian.pkl', 'wb') as f:
                pickle.dump(self.U, f, protocol=pickle.HIGHEST_PROTOCOL)
            print('FINISHED')
        else:
            print('HAMILTONIAN FOUND')
            with open('exp_hamiltonian.pkl', 'rb') as f:
                self.U = pickle.load(f)
            assert self.U.shape == (h_shape, h_shape)
        
    def k(self, n):
        """
        Compute the wave number for the nth basis function.
        """
        return np.pi * n / self.L
    
    def phi(self, n):
        """
        Return the nth basis function as a callable.
        
        The basis function is defined as:
            phi_n(x) = sqrt(2/L) * sin(k(n) * x)
        """
        return lambda x: np.sqrt(2 / self.L) * np.sin(self.k(n) * x)
    
    def _psi_from_c_vectorized(self, c):
        """
        Reconstruct the spatial wavefunction from the coefficient vector.
        
        The wavefunction is given by the expansion:
            psi(x) = sum_{n=1}^{h_shape} c_n * phi_n(x)
        where phi_n(x) are the sine basis functions.
        """
        return np.sum(c[:, None] * self.phi_vals, axis=0)

    def _compute_hamiltonian(self):
        """
        Compute the Hamiltonian matrix for the chosen basis.
        """
        n_vals = np.arange(1, self.h_shape + 1)
        H = np.diag(self.k(n_vals)**2 / (2 * self.m))
        
        I, J = np.meshgrid(np.arange(self.h_shape), np.arange(self.h_shape), indexing='ij')
        
        def quad_integral(i, j):
            i, j = int(i), int(j)
            return quad(func=lambda x: self.phi(i + 1)(x) * self.phi(j + 1)(x),
                        a=self.x_V, b=self.x_V + self.h)[0]
        
        integrals = np.vectorize(quad_integral)(I, J)
        H += self.V_0 * integrals
        return H
    
    def run(self):
        """
        Evolve the quantum state over time using the precomputed time evolution operator.
        """
        for _ in range(self.K + 1):
            psi = np.array(self._psi_from_c_vectorized(self.c))
            norm_psi = np.sqrt(np.sum(np.abs(psi)**2) * self.dx)
            psi /= norm_psi
            self.history.append(psi)
            self.c = self.U @ self.c