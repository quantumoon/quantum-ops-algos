import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from simulators import DummyQAOASimulator
from tensor_networks import qaoa_tensor_network
from quantum_utils import *
from tqdm.auto import tqdm as tqdma


def experiment_1():
    np.random.seed(42)
    
    num_layers = 2
    params = np.random.uniform(low=0., high=2 * np.pi, size=(num_layers, 2))
    gammas = {0.0: None,
              0.25: None,
              0.75: None,
              1.: None}
    
    k = 3
    num_samples = 5
    qubits = list(range(2, 18 + 1))
    for gamma in tqdma(gammas):
        tn_mean = []
        tn_std = []
        base_mean = []
        base_std = []
        for num_qubits in tqdma(qubits):
            base_times = []
            tn_times = []
            for sample_seed in range(num_samples):
                bitstrings, ids, zz_pairs = random_zz_min_energy(num_qubits, proba=gamma, seed=sample_seed)
                bitstring = bitstrings[0]
                ids = ids[0]
                
                base_sim = DummyQAOASimulator(num_qubits=num_qubits,
                                              num_layers=num_layers,
                                              ids=ids,
                                              params=params)
                start = perf_counter()
                base_sim.run(zz_pairs=zz_pairs)
                base_times.append(perf_counter() - start)
    
                TN_sim = qaoa_tensor_network(num_qubits=num_qubits,
                                             params=params,
                                             zz_pairs=zz_pairs,
                                             bitstring=bitstring)
                _, tn_time = TN_sim.contract_greedy(k=k, timing=True)
                tn_times.append(tn_time)
            
            tn_mean.append(np.mean(tn_times))
            tn_std.append(np.std(tn_times))
            base_mean.append(np.mean(base_times))
            base_std.append(np.std(base_times))
            
        gammas[gamma] = {'tn': [qubits, tn_mean, tn_std], 'base': [qubits, base_mean, base_std]}
    
    fig, (ax_tn, ax_base) = plt.subplots(1, 2, figsize=(10, 4), dpi=300, sharey=True)
    for gamma, data in gammas.items():
        x_tn, y_tn, err_tn = data['tn']
        x_b,  y_b,  err_b  = data['base']
        
        ax_tn.errorbar(x_tn, y_tn, err_tn, marker='o', linestyle='-', 
                       label=f'γ={gamma}', alpha=0.7)
        ax_base.errorbar(x_b, y_b, err_b, marker='s', linestyle='--',
                         label=f'γ={gamma}')
        
    fig.suptitle(f'Number of layers = {num_layers}, number of greedy steps = {k}', fontsize=14, y=1.02)
    
    ax_tn.set_title('Tensor Network')
    ax_tn.set_xlabel('Number of qubits')
    ax_tn.set_ylabel('Simulation time')
    ax_tn.set_yscale('log')
    ax_tn.legend()
    ax_tn.grid(True)
    
    ax_base.set_title('Circuit')
    ax_base.set_xlabel('Number of qubits')
    ax_base.set_yscale('log')
    ax_base.legend()
    ax_base.grid(True)
    
    plt.tight_layout()
    plt.show()


def experiment_2():
    np.random.seed(42)
    
    num_layers = 2
    params = np.random.uniform(low=0., high=2 * np.pi, size=(num_layers, 2))
    
    k = 3
    num_samples = 5
    num_qubits = 10
    gammas = np.arange(0, 1+0.1, 0.1)
    tn_mean = []
    tn_std = []
    base_mean = []
    base_std = []
    for gamma in tqdma(gammas):
        base_times = []
        tn_times = []
        for sample_seed in range(num_samples):
            bitstrings, ids, zz_pairs = random_zz_min_energy(num_qubits, proba=gamma, seed=sample_seed)
            bitstring = bitstrings[0]
            ids = ids[0]
            
            base_sim = DummyQAOASimulator(num_qubits=num_qubits,
                                          num_layers=num_layers,
                                          ids=ids,
                                          params=params)
            start = perf_counter()
            base_sim.run(zz_pairs=zz_pairs)
            base_times.append(perf_counter() - start)
    
            TN_sim = qaoa_tensor_network(num_qubits=num_qubits,
                                         params=params,
                                         zz_pairs=zz_pairs,
                                         bitstring=bitstring)
            _, tn_time = TN_sim.contract_greedy(k=k, timing=True)
            tn_times.append(tn_time)
            
        tn_mean.append(np.mean(tn_times))
        tn_std.append(np.std(tn_times))
        base_mean.append(np.mean(base_times))
        base_std.append(np.std(base_times))

    plt.figure(dpi=300)
    plt.errorbar(gammas, tn_mean, tn_std, label='Tensor Network', marker='o')
    plt.errorbar(gammas, base_mean, base_std, label='Circuit', linestyle='dashed', marker='s')
    plt.grid(True)
    plt.ylabel('Simulation time, s')
    plt.xlabel('$\\gamma$')
    plt.title(f'Number of qubits = {num_qubits}, number of layers = {num_layers}')
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    print('Starting experiment №1')
    experiment_1()
    print('Starting experiment №2')
    experiment_2()
    print('Done')