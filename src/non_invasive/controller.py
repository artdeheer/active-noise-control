from scipy.linalg import toeplitz
import numpy as np


class NonInvasiveController:
    def __init__(self, m):
        self.m = m
        self.g_weights = np.zeros(self.m)
        self.r_buffer = np.zeros(self.m)

    def solve_or_update(self, h_hat, s_hat):
        """Solve the direct FIR normal equation R_s g = -p."""
        h_hat = np.asarray(h_hat, dtype=float)
        s_hat = np.asarray(s_hat, dtype=float)

        if np.linalg.norm(s_hat) < 1e-12:
            return

        # Equation (29): Toeplitz autocorrelation matrix of the secondary path.
        r_elements = np.array(
            [np.dot(s_hat[: self.m - k], s_hat[k:]) for k in range(self.m)],
            dtype=float,
        )
        Rs = toeplitz(r_elements)

        # Equation (28): p = M_s^T h_hat.
        p = np.array(
            [np.dot(s_hat[: self.m - k], h_hat[k:]) for k in range(self.m)],
            dtype=float,
        )

        try:
            self.g_weights = np.linalg.solve(Rs, -p)
        except np.linalg.LinAlgError:
            self.g_weights = -np.linalg.pinv(Rs) @ p

    def generate_actuation(self, r_n):
        self.r_buffer[1:] = self.r_buffer[:-1]
        self.r_buffer[0] = r_n

        return float(np.dot(self.r_buffer, self.g_weights))
