import numpy as np
import matplotlib.pyplot as plt
from simulators import *
from tqdm.auto import tqdm as tqdma
from scipy.stats import entropy

num_qubits = 8
sigma = 0.2
p = 5

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
    P[0, i] = entropy(p_1, p_true)
    
    p_2 = approx_simulation(num_qubits=num_qubits,
                            num_layers=p,
                            num_shots=num_shots,
                            sigma=sigma,
                            mode='Choosing from 2 Kraus ops')
    P[1, i] = entropy(p_2, p_true)

plt.figure(dpi=150)
labels = [f'Sampling noise parameter $\\delta$ from N(0.0, {sigma})', 'Choosing from 2 Kraus ops']
for i in range(2):
    plt.plot(X, P[i], label=labels[i], marker='.', alpha=0.7)
plt.legend()
plt.xlabel('Number of shots')
plt.ylabel('KL divergence')
plt.title('KL divergence between true and approximate distributions')
plt.hlines(0, 0, 1100, linestyle='--', color='r')
plt.xlim(0, X[-1] + 5)
plt.savefig('KLD.png')