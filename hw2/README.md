# Spin Echo Simulation

This folder contains a simulation of a spin echo experiment for a single-qubit system interacting with an environment of N qubits.

## Assignment Description

Consider a system composed of one qubit that interacts with N environmental qubits. The interaction Hamiltonian is defined as:

$$
H = \sum_{n=1}^{N} \gamma_n Z_{\text{sys}} \otimes Z_{\text{env}, n}
$$

where:
- $$Z_{\text{sys}}$$ is the Pauli Z operator acting on the system qubit.
- $$Z_{\text{env}, n}$$ is the Pauli Z operator acting on the $$n$$-th environmental qubit.
- $$\gamma_n$$ are the coupling constants.

The initial state of the system is $$|+\rangle$$, and the initial state of the environment is given by a Gibbs distribution.

A $$\pi$$-pulse (corresponding to the spin echo operation) is applied at time $$\tau$$.

The task is to simulate the dynamics of the system over the time interval from $$0$$ to $$2\tau$$ and to plot the following:
- **Purity** of the system’s quantum state as a function of time.
- **Fidelity** $$F(\rho_{\text{initial}}, \rho_{\text{current}})$$ as a function of time.
