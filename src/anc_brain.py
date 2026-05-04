import numpy as np
import config as cfg
from algorithms.invasive import InvasiveModel
from algorithms.non_invasive import NonInvasiveModel

class ANCBrain:
    def __init__(self, method='non-invasive'):
        # 1. Select the Modeling Strategy (Engine A)
        if method == 'invasive':
            self.modeling_engine = InvasiveModel(cfg.S_TAPS, cfg.MU_MODELING)[cite: 1]
        else:
            self.modeling_engine = NonInvasiveModel(cfg.S_TAPS)[cite: 1]
        
        self.method = method
        
        # 2. Controller Weights (Engine B)
        # These are the 'g' or 'h' coefficients that create anti-noise
        self.h_weights = np.zeros(cfg.M_TAPS) 
        
        # 3. Data Buffers (Circular/Sliding Windows)
        # Reference noise buffer x(n)
        self.x_buffer = np.zeros(cfg.M_TAPS)
        # Filtered-x buffer (x filtered through S_hat)
        self.fx_buffer = np.zeros(cfg.M_TAPS) 

    def process_sample(self, x_n, e_n):
        """
        The core real-time execution step called every sample.
        """
        # A. Update Reference Buffer
        # Push the new noise sample x_n into self.x_buffer
        
        # B. Modeling (Engine A)
        # Call self.modeling_engine.update() to refine the 'Digital Twin' S_hat[cite: 1]
        
        # C. Control Calculation (Engine B)
        # 1. Compute anti-noise: y_n = dot_product(h_weights, x_buffer)
        # 2. If invasive, add generated white noise to y_n[cite: 1]
        
        # D. Weight Update (FxLMS)
        # 1. Generate Filtered-x: x_filtered = dot_product(S_hat, x_buffer_subset)
        # 2. Update h_weights: h = h + mu * e_n * x_filtered
        
        return y_n