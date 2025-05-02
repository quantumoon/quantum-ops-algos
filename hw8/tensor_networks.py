import numpy as np
from simulators import ContractionGraph
from quantum_utils import rzz, rx
from typing import Tuple, List


def qaoa_tensor_network(num_qubits: int,
                        params: np.ndarray,
                        zz_pairs: List[Tuple[int, int]],
                        bitstring: List[int]) -> ContractionGraph:
    
    plus = np.array([1, 1]) / 2**0.5
    basis = [np.array([1, 0]),
             np.array([0, 1])]
    
    g = ContractionGraph(num_qubits=num_qubits)
    
    for q in range(num_qubits):
        g.add_node(plus, f'$+_{q}$', q)
    
    for i, (gamma, beta) in enumerate(params):
        RZZ = rzz(gamma)
        RX = rx(beta)
        for q_up, q_down in zz_pairs:
            g.add_node(RZZ, f'rzz_{q_up}{q_down}_{i}', (q_up, q_down))
        for q in range(num_qubits):
            g.add_node(RX, f'rx_{q}_{i}', q)
    
    for q, bit in enumerate(bitstring):
        g.add_node(basis[bit], f'${bit}_{q}$', q)
    
    return g