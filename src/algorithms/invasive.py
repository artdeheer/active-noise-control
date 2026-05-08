import numpy as np
import config as cfg

class InvasiveEngine:
    def __init__(self):
        # The estimate of the secondary path
        self.s_hat = np.zeros(cfg.S_TAPS)
        
        # Buffer for probe signal (white noise) history
        self.v_buffer = np.zeros(cfg.S_TAPS)

    def update(self, x_n, e_n, y_n):
        """
        Generate probe noise and update secondary path estimate.
        Returns: (s_hat, v_n) - estimated path and probe signal
        """
        # Generate probe signal (white noise for identification)
        v_n = np.random.normal(0, cfg.V_NOISE_AMP)
        
        # Update probe buffer
        self.v_buffer[1:] = self.v_buffer[:-1]
        self.v_buffer[0] = v_n
        
        # Model the secondary path using probe signal
        # Prediction: what the error should be based on current s_hat estimate
        e_modeling = e_n - np.dot(self.v_buffer, self.s_hat)
        
        # Adapt s_hat using LMS
        self.s_hat += cfg.MU_MODELING * e_modeling * self.v_buffer
        
        return self.s_hat, v_n