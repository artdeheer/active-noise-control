import numpy as np
import config as cfg

class ANCBrain:
    def __init__(self, modeling_engine):
        # 1. The Adaptive Filter (H weights)
        # This is the brain's internal 'counter-strategy'
        self.h_weights = np.zeros(cfg.M_TAPS) 
        
        # 2. Reference Buffers
        self.x_buffer = np.zeros(cfg.M_TAPS)
        
        # 3. The Modeling Engine (The 'Invasive' or 'Non-Invasive' logic)
        self.modeling_engine = modeling_engine

    def process_sample(self, x_n, e_n):
        # STEP 1: Update Reference Buffer FIRST
        self.x_buffer[1:] = self.x_buffer[:-1]
        self.x_buffer[0] = x_n

        # STEP 2: Generate Anti-Noise
        y_n = np.dot(self.x_buffer, self.h_weights)

        # STEP 3: Get the latest path guess (s_hat) from the engine
        # For Invasive, this also gives us the white noise v_n
        s_hat, v_n = self.modeling_engine.update(x_n, e_n, y_n)

        # STEP 4: Filtered-x (The most important part)
        # We must filter x through s_hat to align the timing
        # We only take the first len(s_hat) samples from x_buffer
        x_filtered = np.dot(self.x_buffer[:len(s_hat)], s_hat)

        # STEP 5: Weight Update
        # If e_n is positive and x_filtered is positive, 
        # we subtract to reduce the noise.
        # Calculate power of the filtered signal to stabilize learning
        reg = 1e-6  # small number to prevent division by zero
        x_pow = np.dot(x_filtered, x_filtered) + reg

        # Normalized update
        self.h_weights -= (cfg.MU_CONTROLLER / x_pow) * e_n * x_filtered

        return y_n + v_n