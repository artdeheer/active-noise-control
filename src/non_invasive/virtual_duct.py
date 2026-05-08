import numpy as np
import config as cfg

class VirtualDuct:
    def __init__(self):
        self.buffer_P = np.zeros(cfg.PRIMARY_DELAY_SAMPLES)
        self.filter_len = 10 
        self.buffer_S = np.zeros(self.filter_len + cfg.SECONDARY_DELAY_SAMPLES)
        
        # Initial Physics (The "Guessing" Target)
        self.L = self.filter_len
        self.generate_s_truth(damping=0.1, freq=0.1, gain=1.0)

    def simulate_step(self, x_n, y_n):
        # PRIMARY PATH 
        d_n_pure = self.buffer_P[-1]
        d_n = d_n_pure * 0.95 
        
        self.buffer_P[1:] = self.buffer_P[:-1]
        self.buffer_P[0] = x_n

        # SECONDARY PATH
        self.buffer_S[1:] = self.buffer_S[:-1]
        self.buffer_S[0] = y_n

        delay = cfg.SECONDARY_DELAY_SAMPLES
        arrival_window = self.buffer_S[delay : delay + self.L]
        
        y_prime = np.dot(arrival_window, self.s_truth)
        e_n = d_n + y_prime

        return d_n, e_n
    
    def generate_s_truth(self, damping, freq, gain=1.0):
        """
        Internal helper to build the secondary path impulse response.
        """
        n_axis = np.arange(self.L)
        decay_envelope = np.exp(-damping * n_axis)
        oscillation = np.sin(2 * np.pi * freq * n_axis)
        
        # We apply the gain here to control the 'volume' of the secondary path
        self.s_truth = gain * (decay_envelope * oscillation)
    def change_environment(self, damping=0.05, freq=0.15, gain=-1.0):
        """
        Updates the physics of the duct.
        - damping: How fast the ring settles.
        - freq: The resonant frequency (speed of sound/temp).
        - gain: Overall volume and phase (negative = phase flip).
        """
        print(f"--- Physical Environment Changed: freq={freq}, gain={gain} ---")
        self.generate_s_truth(damping, freq, gain)