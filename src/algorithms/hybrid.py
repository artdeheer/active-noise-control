import numpy as np
import config as cfg

class HybridEngine:
    def __init__(self, threshold=0.01):
        # FIX: Use S_TAPS for the size, not the Learning Rate!
        self.filter_len = cfg.S_TAPS 
        
        self.s_hat = np.zeros(self.filter_len) 
        self.y_buffer = np.zeros(self.filter_len)
        self.v_buffer = np.zeros(self.filter_len)
        
        # Learning rates pulled from config
        self.mu_passive = cfg.MU_MODELING      
        self.mu_active = cfg.MU_MODELING * 5   # Active modeling can usually be much faster
        self.threshold = threshold
        self.is_active = False

    def update(self, x_n, e_n, y_n):
        """
        Hybrid Engine: Passive by default, Active when environment shifts.
        """
        # 1. Slide the buffers
        self.y_buffer[1:] = self.y_buffer[:-1]
        self.y_buffer[0] = y_n

        # 2. Calculate current modeling error (Passive)
        # Prediction: What should e_n be based on our current pipe map?
        prediction = np.dot(self.y_buffer, self.s_hat)
        e_modeling_passive = e_n - prediction

        # 3. Decision Logic: Trigger 'Active' mode if the error is too high
        # This happens if someone moves the mic or the pipe temperature changes
        if np.abs(e_modeling_passive) > self.threshold:
            self.is_active = True
            v_n = np.random.normal(0, 0.05) # Small hiss of white noise
        else:
            self.is_active = False
            v_n = 0.0

        # 4. Update the Active (Probe) Buffer
        self.v_buffer[1:] = self.v_buffer[:-1]
        self.v_buffer[0] = v_n

        # 5. Dual-Update Step (The LMS Part)
        if self.is_active:
            # Active Update: Use the white noise 'fingerprint' to fix the map fast
            # We add back the v_n history to the calculation
            e_modeling_active = e_n - (prediction + np.dot(self.v_buffer, self.s_hat))
            self.s_hat += self.mu_active * e_modeling_active * self.v_buffer
        else:
            # Passive Update: Subtle adjustments while the system is working well
            self.s_hat += self.mu_passive * e_modeling_passive * self.y_buffer

        return self.s_hat, v_n