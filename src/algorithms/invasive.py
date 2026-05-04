from .base_model import SecondaryPathModel
import numpy as np

class InvasiveModel(SecondaryPathModel):
    def __init__(self, s_taps, mu_m):
        super().__init__(s_taps)
        self.mu_m = mu_m # Step size for modeling
        self.v_history = np.zeros(s_taps) # History of injected noise

    def generate_noise(self, amplitude):
        """Returns a white noise sample v(n) to be played."""
        return np.random.normal(0, amplitude)

    def update(self, error_sample, ref_sample, v_n):
        """
        Math: s_hat = s_hat + mu_m * error * v_history[cite: 1]
        Focuses purely on the correlation between white noise and the mic.[cite: 1]
        """
        pass