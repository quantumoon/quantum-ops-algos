import numpy as np

class CliffordSimulator:
    def __init__(self, n):
        self.n = n
        self.tableau = np.zeros((2 * n, 2 * n + 1), dtype=np.uint8)
        # X_i
        for i in range(n):
            self.tableau[i, i] = 1
        # Z_i
        for i in range(n):
            self.tableau[n + i, n + i] = 1

    def apply_H(self, q):
        for r in range(2 * self.n):
            x_q = self.tableau[r, q]
            z_q = self.tableau[r, self.n + q]
            self.tableau[r, q] = z_q
            self.tableau[r, self.n + q] = x_q
            if x_q == 1 and z_q == 1:
                self.tableau[r, 2 * self.n] ^= 1

    def apply_CNOT(self, control, target):
        for r in range(2 * self.n):
            if self.tableau[r, control] == 1 and self.tableau[r, self.n + target] == 1:
                self.tableau[r, 2 * self.n] ^= 1
            self.tableau[r, target] ^= self.tableau[r, control]
            self.tableau[r, self.n + control] ^= self.tableau[r, self.n + target]

    def tableau_to_stabilizers(self):
        pauli_map = {
            (0, 0): 'I',
            (1, 0): 'X',
            (1, 1): 'Y',
            (0, 1): 'Z'
        }
        stabilizers = []
        for r in range(self.n, 2 * self.n):
            sign = '-' if self.tableau[r, 2 * self.n] == 1 else ''
            op_str = ""
            for q in range(self.n):
                x = self.tableau[r, q]
                z = self.tableau[r, self.n + q]
                op_str += pauli_map[(x, z)]
            stabilizers.append(sign + op_str)
        return stabilizers

    def measure(self, q):
        # If there are any X stabilizers, then measurement is not deterministic
        # Otherwise it is
        anti_row = None
        for r in range(self.n, 2 * self.n):
            if self.tableau[r, q] == 1:
                anti_row = r
                break
        
        # Deterministic
        if anti_row is None:
            for r in range(self.n, 2 * self.n):
                if self.tableau[r, q] == 0 and self.tableau[r, self.n + q] == 1:
                    return int(self.tableau[r, 2 * self.n])
            return 0
        
        # Not deterministic
        result = np.random.randint(2)
        for r in range(2 * self.n):
            if r != anti_row and self.tableau[r, q] == 1:
                self.tableau[r, :] ^= self.tableau[anti_row, :]
        
        self.tableau[anti_row, :] = 0
        self.tableau[anti_row, self.n + q] = 1
        self.tableau[anti_row, 2 * self.n] = result
        
        self.tableau[q, :] = 0
        self.tableau[q, q] = 1
        self.tableau[q, 2 * self.n] = 0
        return result