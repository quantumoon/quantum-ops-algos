import numpy as np


def apply_1qubit_gate(state, U, i):
    result = np.tensordot(U, state, axes=([1], [i]))
    result = np.moveaxis(result, 0, i)
    return result


def apply_2qubit_gate(state, U, i, j):
    if U.shape == (4, 4):
        U = U.reshape(2, 2, 2, 2)
    result = np.tensordot(U, state, axes=([2, 3], [i, j]))
    result = np.moveaxis(result, [0, 1], [i, j])
    return result


def apply_1qubit_superop(rho, G, i):
    N = rho.ndim // 2
    if G.shape == (4, 4):
        G = G.reshape(2, 2, 2, 2)
    result = np.tensordot(G, rho, axes=[[2, 3], [i, i + N]])
    result = np.moveaxis(result, [0, 1], [i, i + N])
    return result


def apply_2qubit_superop(rho, G, i, j):
    N = rho.ndim // 2
    if G.shape == (16, 16):
        G = G.reshape([2] * 8)
    result = np.tensordot(G, rho, axes=[[4, 5, 6, 7], [i, j, i + N, j + N]])
    result = np.moveaxis(result, [0, 1, 2, 3], [i, j, i + N, j + N])
    return result


def explicit_simulation(num_qubits, num_layers, sigma):
    G_H = 0.5 * np.array([[1, 1,1 ,1],
                          [1,-1,1,-1],
                          [1,1,-1,-1],
                          [1,-1,-1,1]], dtype=np.complex64)
    G_CZ = np.diag([1,1,1,-1,
                    1,1,1,-1,
                    1,1,1,-1,
                    -1,-1,-1,1]).astype(np.complex64)
    G_noise = (0.5*(1+np.exp(-sigma**2/2)) * np.eye(4, dtype=np.complex64) + \
               0.5*(1-np.exp(-sigma**2/2)) * np.eye(4, dtype=np.complex64)[::-1])
    
    rho = np.zeros((1<<num_qubits, 1<<num_qubits), dtype=np.complex64)
    rho[0, 0] = 1
    rho = rho.reshape([2, 2] * num_qubits)
    
    for _ in range(num_layers):
        for n in range(num_qubits):
            rho = apply_1qubit_superop(rho, G_H, n)
            rho = apply_1qubit_superop(rho, G_noise, n)
        
        for n in range(num_qubits):
            rho = apply_2qubit_superop(rho, G_CZ, n, (n+1) % num_qubits)
    
    return np.diagonal(rho.reshape((1<<num_qubits, 1<<num_qubits))).real


def approx_simulation(num_qubits, num_layers, num_shots, sigma, mode, seed=42):
    assert mode in ['Sampling noise parameter', 'Choosing from 2 Kraus ops']
    sampling = (mode == 'Sampling noise parameter')
    
    np.random.seed(seed)
    if not sampling:
        Kraus = [np.eye(2, dtype=np.complex64),
                 np.eye(2, dtype=np.complex64)[::-1]]
        p = [0.5*(1+np.exp(-sigma**2/2)),
             0.5*(1-np.exp(-sigma**2/2))]
    
    H = 2**-0.5 * np.array([[1, 1],
                            [1,-1]], dtype=np.complex64)
    CZ = np.diag([1, 1, 1, -1]).astype(np.complex64)
    R_x = lambda t: np.array([[np.cos(t/2), -1j*np.sin(t/2)],
                              [-1j*np.sin(t/2), np.cos(t/2)]])
    

    probs = np.zeros(1<<num_qubits)
    for shot in range(num_shots):
        state = np.zeros(1<<num_qubits, dtype=np.complex64)
        state[0] = 1
        state = state.reshape([2] * num_qubits)
        for _ in range(num_layers):
            for n in range(num_qubits):
                state = apply_1qubit_gate(state, H, n)
                if sampling:
                    theta = sigma * np.random.randn(1)[0]
                    state = apply_1qubit_gate(state, R_x(theta), n)
                else:
                    U = Kraus[np.random.choice([0, 1], p=p)]
                    state = apply_1qubit_gate(state, U, n)
            
            for n in range(num_qubits):
                state = apply_2qubit_gate(state, CZ, n, (n + 1) % num_qubits)
        state /= np.sqrt(np.sum(np.abs(state)**2))
        i = np.random.choice(np.arange(1<<num_qubits), p=np.abs(state.reshape(1<<num_qubits))**2)
        probs[i] += 1
    return probs / num_shots