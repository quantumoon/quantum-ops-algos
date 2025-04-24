import numpy as np
from typing import List
from utils import *


class SmartQAOASimulator:
    def __init__(self,
                 num_qubits: int,
                 out_bitstrings: List[List[int]],
                 params: List[float]):
        self.g1, self.g2, self.b1, self.b2 = params
        self.num_qubits = num_qubits
        if len(out_bitstrings[0]) != num_qubits:
            raise ValueError()
        self.out_bitstrings = out_bitstrings

    def collapse_1q(self, psi_out):
        psi = np.array([1,
                        1], dtype=complex) / 2**0.5
        rx1 = rx(self.b1)
        rx2 = rx(self.b2)
        
        # first layer
        for i in range(1, self.num_qubits):
            self.layer1[i - 1] = np.tensordot(psi, self.layer1[i - 1], axes=([0], [-1]))
            self.layer1[i - 1] = np.tensordot(rx1, self.layer1[i - 1], axes=([-1], [0]))
        self.layer1[0] = np.tensordot(psi, self.layer1[0], axes=([0], [-1]))
        self.layer1[-1] = np.moveaxis(np.tensordot(rx1, self.layer1[-1], axes=([-1], [1])), 0, 1)

        # second layer
        for i in range(1, self.num_qubits):
            self.layer2[i - 1] = np.tensordot(rx2, self.layer2[i - 1], axes=([1], [0]))[psi_out[i - 1], ...]
        self.layer2[-1] = np.tensordot(rx2, self.layer2[-1], axes=([1], [0]))[psi_out[-1], ...]
        
        self.layer1 = self.layer1[::-1]
        self.layer2 = self.layer2[::-1]
        
    def collapse_layers(self):
        U, S = self.layer1.pop(), self.layer2.pop()
        if self.layer1:
            U_ = np.tensordot(U, S, axes=([-2], [-2]))
            self.layer1[-1] = np.tensordot(U_, self.layer1[-1], axes=([0, -1], [-1, 0]))
            return self.collapse_layers()
        return np.tensordot(U, S, axes=([0, 1], [0, 1]))
        
    def run(self):
        p = 0.
        for psi_out in self.out_bitstrings:
            self.layer1 = [rzz(self.g1) for _ in range(self.num_qubits - 1)]
            self.layer2 = [rzz(self.g2) for _ in range(self.num_qubits - 1)]
            
            self.collapse_1q(psi_out)
            p += np.abs(self.collapse_layers())**2
        return p


class DummyQAOASimulator:
    def __init__(self,
                 num_qubits: int,
                 ids: List[int],
                 params: List[float]):
        self.num_qubits = num_qubits
        self.out_ids = ids
        self.g1, self.g2, self.b1, self.b2 = params
        self.state = np.ones([2] * num_qubits, dtype=complex) / 2**(num_qubits/2)

    def _apply_1(self, U, i: int):
        self.state = np.tensordot(U, self.state, axes=([1], [i]))
        self.state = np.moveaxis(self.state, 0, i)

    def _apply_2(self, U: List, i: int, j: int):
        if U.ndim == 4:
            U = U.reshape(2,2,2,2)
        self.state = np.tensordot(U, self.state, axes=([2, 3], [i, j]))
        self.state = np.moveaxis(self.state, [0, 1], [i, j])

    def run(self):
        rzz1, rzz2 = rzz(self.g1), rzz(self.g2)
        rx1, rx2 = rx(self.b1), rx(self.b2)
        for q in range(self.num_qubits - 1):
            self._apply_2(rzz1, q, q + 1)
        for q in range(self.num_qubits):
            self._apply_1(rx1, q)
        for q in range(self.num_qubits - 1):
            self._apply_2(rzz2, q, q + 1)
        for q in range(self.num_qubits):
            self._apply_1(rx2, q)
        ground_probs = np.abs(self.state.reshape(-1)[self.out_ids])**2
        return ground_probs.sum()