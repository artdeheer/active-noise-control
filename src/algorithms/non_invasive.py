import numpy as np
import config as cfg

class NonInvasiveEngine:
    def __init__(self):
        # 1. The 'Guess' of the secondary path
        # Length 10 to match your 's_truth' in VirtualDuct
        self.s_hat = np.zeros(10) 
        
        # 2. Buffer for the anti-noise y_n
        # This is what Engine A uses to see how the speaker behaves
        self.y_buffer = np.zeros(10)

    def update(self, x_n, e_n, y_n):
        """
        Engine A: Updates the s_hat estimate using only existing signals.
        """
        # --- STEP A: Update the history of what we played ---
        self.y_buffer[1:] = self.y_buffer[:-1]
        self.y_buffer[0] = y_n

        # --- STEP B: Calculate the 'Modeling Error' ---
        # We check: does our current s_hat correctly predict e_n?
        # Note: In non-invasive, we treat e_n as the target for our y_n history
        e_modeling = e_n - np.dot(self.y_buffer, self.s_hat)

        # --- STEP C: Update the s_hat weights (LMS style) ---
        # We nudge s_hat so it gets better at predicting how y_n becomes e_n
        # mu_s is the modeling learning rate (usually much smaller than mu_h)
        self.s_hat += cfg.MU_MODELING * e_modeling * self.y_buffer

        return self.s_hat, 0.0