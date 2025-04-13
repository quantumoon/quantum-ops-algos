from clifford_simulator import CliffordSimulator

n = 10
sim = CliffordSimulator(n)

sim.apply_H(0)
for q in range(n - 1):
    sim.apply_CNOT(q, q + 1)

pauli_stabilizers = sim.tableau_to_stabilizers()
for stab in pauli_stabilizers:
    print(stab)