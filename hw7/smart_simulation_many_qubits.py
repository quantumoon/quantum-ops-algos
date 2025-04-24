import time
import numpy as np
import matplotlib.pyplot as plt
from simulators import *
from utils import zz_hamiltonian_eigens

np.random.seed(42)
params = np.random.uniform(0., 2 * np.pi, size=4)

range_smart = range(2, 1000 + 25, 25)
n_runs = 10

smart_means = []
smart_stds  = []

for n in range_smart:
    out_bstr, _ = zz_hamiltonian_eigens(n, compute_honestly=False)

    times = []
    for _ in range(n_runs):
        sim = SmartQAOASimulator(num_qubits=n,
                                 out_bitstrings=out_bstr,
                                 params=params)
        t0 = time.perf_counter()
        _ = sim.run()
        times.append(time.perf_counter() - t0)

    smart_means.append(np.mean(times))
    smart_stds.append(np.std(times))

plt.figure(figsize=(8, 5), dpi=200)

plt.errorbar(range_smart, smart_means, yerr=smart_stds,
             marker='o', linestyle='-', label='SmartQAOA')
plt.xlabel('Number of qubits')
plt.ylabel('SmartQAOA circuit execution time, s')
plt.legend()
plt.grid(True, linestyle=':')
plt.tight_layout()
plt.savefig('TN_simulation.png')
plt.show()