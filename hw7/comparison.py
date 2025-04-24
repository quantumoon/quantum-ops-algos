import time
import numpy as np
import matplotlib.pyplot as plt
from simulators import *
from utils import zz_hamiltonian_eigens

np.random.seed(42)
params = np.random.uniform(0., 2 * np.pi, size=4)

qubit_range = range(2, 18 + 1)
n_runs = 10

tn_means = []
tn_stds  = []
dm_means = []
dm_stds  = []

for n in qubit_range:
    out_bitstrings, ids = zz_hamiltonian_eigens(n)

    tn_times = []
    dm_times = []

    for _ in range(n_runs):
        tn_sim = SmartQAOASimulator(num_qubits=n,
                                    out_bitstrings=out_bitstrings,
                                    params=params)
        t0 = time.perf_counter()
        _ = tn_sim.run()
        tn_times.append(time.perf_counter() - t0)

        dm_sim = DummyQAOASimulator(num_qubits=n,
                                    ids=ids,
                                    params=params)
        t0 = time.perf_counter()
        _ = dm_sim.run()
        dm_times.append(time.perf_counter() - t0)

    tn_means.append(np.mean(tn_times))
    tn_stds.append(np.std(tn_times))
    dm_means.append(np.mean(dm_times))
    dm_stds.append(np.std(dm_times))

plt.figure(figsize=(8, 5), dpi=200)
plt.errorbar(qubit_range, tn_means, yerr=tn_stds,
             marker='o', linestyle='-', label='SmartQAOA')
plt.errorbar(qubit_range, dm_means, yerr=dm_stds,
             marker='s', linestyle='--', label='DummyQAOA')
plt.xlabel('Number of qubits')
plt.ylabel('QAOA circuit execution time, s')
plt.yscale('log')
plt.title('Two simulation strategies execution time comparison')
plt.legend()
plt.grid(True, linestyle=':')
plt.tight_layout()
plt.savefig('comparison.png')
plt.show()