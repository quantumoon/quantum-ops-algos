import numpy as np
from simulators import *
from utils import *
from tqdm.auto import tqdm as tqdma

np.random.seed(42)
params = np.random.uniform(low=0., high=2 * np.pi, size=4)

print('Comparison of simulation results for 2 simulators')
print('Starting comparison...\n')
for num_qubits in tqdma(range(2, 12 + 1)):
    out_bitstrings, ids = zz_hamiltonian_eigens(num_qubits)
    
    tn_sim = SmartQAOASimulator(num_qubits=num_qubits,
                                out_bitstrings=out_bitstrings,
                                params=params)
    dm_sim = DummyQAOASimulator(num_qubits=num_qubits,
                                ids=ids,
                                params=params)
    
    tn_p = tn_sim.run()
    dm_p = dm_sim.run()
    assert np.allclose(tn_p, dm_p)
print('\nAll done')
print('Simulation results are the same')