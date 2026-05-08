from .base_controller import BaseController
import config as cfg
import numpy as np

class InvasiveController(BaseController):
    def __init__(self):
        # Adaptive filter weights for anti-noise generation
        self.h_weights = np.zeros(cfg.M_TAPS)
        
        # Reference signal buffer
        self.x_buffer = np.zeros(cfg.M_TAPS)

    def generate_actuation(self, x_n):
        """
        Generate anti-noise output using current weights and reference signal.
        """
        # Update reference buffer
        self.x_buffer[1:] = self.x_buffer[:-1]
        self.x_buffer[0] = x_n
        
        # Generate anti-noise
        y_n = np.dot(self.x_buffer, self.h_weights)
        return y_n

    def solve_or_update(self, h_hat, s_hat, x_n, e_n):
        """
        Update controller weights using filtered-x LMS.
        """
        # Compute filtered-x signal: filter reference through estimated secondary path
        x_filtered = np.dot(self.x_buffer[:len(s_hat)], s_hat)
        
        # Normalized step size to stabilize learning
        reg = 1e-6  # Small regularization to prevent division by zero
        x_pow = x_filtered * x_filtered + reg
        
        # Filtered-x LMS update
        # If error is positive and filtered input is positive, subtract to reduce error
        self.h_weights -= (cfg.MU_CONTROLLER / x_pow) * e_n * x_filtered