import numpy as np
import config as cfg

class VirtualDuct:
    def __init__(self):
        # Initialize circular buffers based on calculated sample delays
        self.buffer_P = np.zeros(cfg.PRIMARY_DELAY_SAMPLES)
        self.buffer_S = np.zeros(cfg.SECONDARY_DELAY_SAMPLES)
        
        # 'Ground Truth' Secondary Path: The math your brain must learn
        # We model this as a small FIR filter representing speaker response
        self.s_truth = np.exp(-0.1 * np.arange(10)) * np.sin(2 * np.pi * 0.1 * np.arange(10))

        self.L = len(self.s_truth)
    def simulate_step(self, x_n, y_n):
        """
        Input: x_n (source noise), y_n (anti-noise from speaker)
        Output: d_n (delayed noise at mic), e_n (error mic reading)
        """

        # PRIMARY PATH
        d_n_pure = self.buffer_P[-1]
        d_n = d_n_pure * 0.95 # sound dissapates as it travels

        self.buffer_P[1:] = self.buffer_P[:-1] # shift all samples to the right
        self.buffer_P[0] = x_n # insert at start

        # SECONDARY PATH
        self.buffer_S[1:] = self.buffer_S[:-1]
        self.buffer_S[0] = y_n

        y_prime = np.dot(self.buffer_S[:self.L], self.s_truth)
        e_n = d_n + y_prime
        
        return d_n, e_n