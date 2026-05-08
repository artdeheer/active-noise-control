import numpy as np
import config as cfg

class InvasiveEngine:
    def __init__(self):
        # 1. The 'Guess' of the secondary path
        self.s_hat = np.zeros(10)
        
        # 2. Buffer for the SECRET probe signal (White Noise)
        self.v_buffer = np.zeros(10)

    def update(self, x_n, e_n, y_n):
        """
        Engine A: Updates s_hat by injecting white noise.
        """
        # --- STEP A: Generate the 'Probe' ---
        # A tiny bit of white noise (e.g., volume 0.01)
        v_n = np.random.normal(0, 0.01)

        # --- STEP B: Update Probe History ---
        self.v_buffer[1:] = self.v_buffer[:-1]
        self.v_buffer[0] = v_n

        # --- STEP C: Calculate Modeling Error ---
        # We check how well our s_hat predicts the microphone signal
        # using ONLY the probe signal as the reference.
        e_modeling = e_n - np.dot(self.v_buffer, self.s_hat)

        # --- STEP D: Update s_hat ---
        self.s_hat += cfg.MU_MODELING * e_modeling * self.v_buffer

        # CRITICAL: We return s_hat AND the v_n so the Brain 
        # can add it to the speaker output
        return self.s_hat, v_n