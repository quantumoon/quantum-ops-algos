import numpy as np
import matplotlib.pyplot as plt
from simulators import explicit_simulation, approx_simulation
from tqdm.auto import tqdm as tqdma
from scipy.stats import entropy

def compute_KL(num_qubits, p, sigma, shots_range):
    p_true = explicit_simulation(num_qubits=num_qubits, num_layers=p, sigma=sigma)
    P = np.zeros((2, len(shots_range)))
    for i, num_shots in enumerate(tqdma(shots_range, desc=f'qubits={num_qubits}, p={p}, sigma={sigma}')):
        p_1 = approx_simulation(num_qubits=num_qubits, num_layers=p,
                                num_shots=num_shots, sigma=sigma,
                                mode='Sampling noise parameter')
        P[0, i] = entropy(p_true, p_1)
        p_2 = approx_simulation(num_qubits=num_qubits, num_layers=p,
                                num_shots=num_shots, sigma=sigma,
                                mode='Choosing from 2 Kraus ops')
        P[1, i] = entropy(p_true, p_2)
    return P

X = range(1, 500, 25)
sqrt = [1/np.sqrt(x) for x in X]

labels = ['Sampling noise parameter $\\delta$ from N(0, $\\sigma^{2}$)', 
          'Choosing from 2 Kraus ops']

# row 1
sigma_values_row1 = [0.1, 1.0, 10.0]
p_row1 = 4
qubits_row1 = 9

# row 2
p_values_row2 = [2, 5, 8]
sigma_row2 = 1.0
qubits_row2 = 9

# row 3
qubits_values_row3 = [2, 5, 9]
p_row3 = 4
sigma_row3 = 1.0

fig, axes = plt.subplots(3, 3, figsize=(18, 18), dpi=300)

fs_xlabel = 14
fs_ylabel = 14
fs_title = 16
fs_legend = 12

for j, sigma in enumerate(sigma_values_row1):
    P = compute_KL(qubits_row1, p_row1, sigma, X)
    ax = axes[0, j]
    for k in range(2):
        ax.plot(X, P[k], label=labels[k], marker='.', markersize=3)
    ax.plot(X, sqrt, linestyle='--', color='red', label='$1/\\sqrt{N}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of shots', fontsize=fs_xlabel)
    ax.set_ylabel('KL divergence', fontsize=fs_ylabel)
    p_I = round((1 + np.exp(-sigma**2 / 2)) / 2, 4)
    p_X = round((1 - np.exp(-sigma**2 / 2)) / 2, 4)
    subtitle = f'p = {p_row1}, $N_q$ = {qubits_row1}, $\\sigma$ = {sigma}\n$p_I$ = {p_I}, $p_X$ = {p_X}'
    ax.set_title(subtitle, fontsize=fs_title)
    ax.legend(fontsize=fs_legend)

for j, p_val in enumerate(p_values_row2):
    P = compute_KL(qubits_row2, p_val, sigma_row2, X)
    ax = axes[1, j]
    for k in range(2):
        ax.plot(X, P[k], label=labels[k], marker='.', markersize=3)
    ax.plot(X, sqrt, linestyle='--', color='red', label='$1/\\sqrt{N}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of shots', fontsize=fs_xlabel)
    ax.set_ylabel('KL divergence', fontsize=fs_ylabel)
    p_I = round((1 + np.exp(-sigma_row2**2 / 2)) / 2, 4)
    p_X = round((1 - np.exp(-sigma_row2**2 / 2)) / 2, 4)
    subtitle = f'p = {p_val}, $N_q$ = {qubits_row2}, $\\sigma$ = {sigma_row2}\n$p_I$ = {p_I}, $p_X$ = {p_X}'
    ax.set_title(subtitle, fontsize=fs_title)
    ax.legend(fontsize=fs_legend)

for j, qubits_val in enumerate(qubits_values_row3):
    P = compute_KL(qubits_val, p_row3, sigma_row3, X)
    ax = axes[2, j]
    for k in range(2):
        ax.plot(X, P[k], label=labels[k], marker='.', markersize=3)
    ax.plot(X, sqrt, linestyle='--', color='red', label='$1/\\sqrt{N}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of shots', fontsize=fs_xlabel)
    ax.set_ylabel('KL divergence', fontsize=fs_ylabel)
    p_I = round((1 + np.exp(-sigma_row3**2 / 2)) / 2, 4)
    p_X = round((1 - np.exp(-sigma_row3**2 / 2)) / 2, 4)
    subtitle = f'p = {p_row3}, $N_q$ = {qubits_val}, $\\sigma$ = {sigma_row3}\n$p_I$ = {p_I}, $p_X$ = {p_X}'
    ax.set_title(subtitle, fontsize=fs_title)
    ax.legend(fontsize=fs_legend)


plt.tight_layout()
plt.savefig('KLD_grid.png')