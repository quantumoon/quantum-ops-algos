import numpy as np
import matplotlib.pyplot as plt
from simulators import *
from tqdm.auto import tqdm as tqdma
from scipy.stats import entropy

num_qubits = 8
sigma = 1.
p = 10


p_true = explicit_simulation(num_qubits=num_qubits,
                             num_layers=p,
                             sigma=sigma)

X = range(1, 500, 25)
P = np.zeros((2, len(X)))
for i, num_shots in enumerate(tqdma(X)):
    p_1 = approx_simulation(num_qubits=num_qubits,
                            num_layers=p,
                            num_shots=num_shots,
                            sigma=sigma,
                            mode='Sampling noise parameter')
    P[0, i] = entropy(p_true, p_1)
    # P[0, i] = np.linalg.norm(p_true - p_1)
    
    p_2 = approx_simulation(num_qubits=num_qubits,
                            num_layers=p,
                            num_shots=num_shots,
                            sigma=sigma,
                            mode='Choosing from 2 Kraus ops')
    P[1, i] = entropy(p_true, p_2)
    # P[1, i] = np.linalg.norm(p_true - p_2)

plt.figure(dpi=150)
labels = ['Sampling noise parameter $\\delta$ from N(0, $\\sigma^{2}$)', 'Choosing from 2 Kraus ops']
for i in range(2):
    plt.plot(X, P[i], label=labels[i], marker='.', alpha=0.7, markersize=3)
# plt.plot(np.arange(1, 500), (np.arange(1, 500)**-0.5), linestyle='--', color='r', label='$\\frac{1}{\\sqrt{N}}$')
plt.legend()
plt.xlabel('Number of shots')
plt.xscale('log')
plt.yscale('log')
plt.ylabel('KL divergence')
p_I = round((1+np.exp(-sigma**2 / 2))/2, 4)
p_X = round((1-np.exp(-sigma**2 / 2))/2, 4)
subtitle = f'p = {p}, $N_q$ = {num_qubits}, $\\sigma$ = {sigma}, $p_I$ = {p_I}, $p_X$ = {p_X}'
plt.title('KL divergence between true and approximate distributions\n' + subtitle)
plt.savefig('KLD.png')