import numpy as np
from simulators import DummyQAOASimulator
from tensor_networks import qaoa_tensor_network
from quantum_utils import *
from tqdm.auto import tqdm as tqdma

num_layers = 5
np.random.seed(42)
params = np.random.uniform(low=0., high=2 * np.pi, size=(num_layers, 2))

print('Test to compare the simulation results for 2 different simulators')

for num_qubits in tqdma(range(2, 9)):
    bitstrings = [list(map(int, np.binary_repr(i, width=num_qubits))) for i in range(1<<num_qubits)]
    ids = [i for i in range(1<<num_qubits)]
    _, __, zz_pairs = random_zz_max_energy(num_qubits, proba=1., seed=num_qubits)
    
    base_sim = DummyQAOASimulator(num_qubits=num_qubits,
                                  num_layers=num_layers,
                                  ids=ids,
                                  params=params)
    base_probs = base_sim.run(zz_pairs=zz_pairs)
    
    tn_probs = []
    for bitstring in bitstrings:
        TN_sim = qaoa_tensor_network(num_qubits=num_qubits,
                                     params=params,
                                     zz_pairs=zz_pairs,
                                     bitstring=bitstring)
        tn_probs.append(TN_sim.contract_dummy())
    tn_probs = np.array(tn_probs)
    
    assert np.allclose(base_probs, tn_probs), f'Failed for num_qubits = {num_qubits}'
print('Test passed')