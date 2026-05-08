from scipy.linalg import toeplitz
import numpy as np

class NonInvasiveController:
    def __init__(self, m):
        self.m = m
        self.g_weights = np.zeros(self.m)
        self.r_buffer = np.zeros(self.m)

    def solve_or_update(self, h_hat, s_hat, x_n, e_n):
        """Analytical FIR solution (Section IV.B)[cite: 1]."""
        # Build Toeplitz Autocorrelation Matrix Rs (Equation 29)[cite: 1]
        r_elements = np.zeros(self.m)
        for k in range(self.m):
            r_elements[k] = np.sum(s_hat[:self.m-k] * s_hat[k:])
        Rs = toeplitz(r_elements)

        # Build cross-correlation vector p (Equation 28)[cite: 1]
        # p = Ms^T * h_hat[cite: 1]
        p = np.zeros(self.m)
        for i in range(self.m):
            p[i] = np.sum(s_hat[:self.m-i] * h_hat[i:])

        try:
            # Stronger regularization to prevent the 'zero weights' trap[cite: 1]
            reg = 0.05 * np.trace(Rs) / self.m + 1e-4
            self.g_weights = np.linalg.solve(Rs + reg * np.eye(self.m), -p)
        except np.linalg.LinAlgError:
            pass 

    def generate_actuation(self, r_n):
        self.r_buffer[1:] = self.r_buffer[:-1]
        self.r_buffer[0] = r_n
        
        # Generate anti-noise output
        raw_a_n = np.dot(self.r_buffer, self.g_weights)
        return np.clip(raw_a_n, -0.5, 0.5)