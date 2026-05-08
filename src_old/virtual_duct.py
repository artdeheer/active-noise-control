import numpy as np
import config as cfg

class VirtualDuct:
    def __init__(self):
        # 1. PRIMARY PATH BUFFER
        # This must exist for simulate_step to work!
        self.buffer_P = np.zeros(cfg.PRIMARY_DELAY_SAMPLES)
        
        # 2. SECONDARY PATH BUFFER
        # We ensure it's at least as long as our filter (10 taps)
        filter_len = 10
        buffer_S_len = max(filter_len, cfg.SECONDARY_DELAY_SAMPLES)
        self.buffer_S = np.zeros(buffer_S_len)
        
        # 3. SECONDARY PATH FILTER (The Fingerprint)
        self.s_truth = np.exp(-0.1 * np.arange(filter_len)) * np.sin(2 * np.pi * 0.1 * np.arange(filter_len))
        self.L = len(self.s_truth)

    def simulate_step(self, x_n, y_n):
        # --- PRIMARY PATH ---
        # If the error persists, check that the line below 
        # matches the 'self.buffer_P' in __init__ exactly.
        d_n_pure = self.buffer_P[-1]
        d_n = d_n_pure * 0.95
        
        self.buffer_P[1:] = self.buffer_P[:-1]
        self.buffer_P[0] = x_n

        # --- SECONDARY PATH ---
        self.buffer_S[1:] = self.buffer_S[:-1]
        self.buffer_S[0] = y_n
        
        # Using the L we defined in __init__
        y_prime = np.dot(self.buffer_S[:self.L], self.s_truth)

        # --- SUMMATION ---
        e_n = d_n + y_prime

        return d_n, e_n