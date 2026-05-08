import numpy as np
import config as cfg

class ANCBrain:
    def __init__(self, modeling_engine, controller, is_orthogonal=False):
        self.modeling_engine = modeling_engine
        self.controller = controller
        self.is_orthogonal = is_orthogonal
        self.last_actuation = 0.0

    def process_sample(self, x_n, e_n):
        """
        Coordinates the adaptation process for both invasive and non-invasive modes.
        Returns: (y_out, v_probe) - actuation signal and probe noise for logging
        """
        if self.is_orthogonal:
            # --- Non-Invasive (Orthogonal) Path ---
            # 1. Update Path Models using last actuation
            h_hat, s_hat = self.modeling_engine.update(x_n, self.last_actuation, e_n)
            
            # 2. Update the Controller
            self.controller.solve_or_update(h_hat, s_hat, x_n, e_n)
            
            # 3. Generate Actuation (no probing for orthogonal)
            y_out = self.controller.generate_actuation(x_n)
            self.last_actuation = y_out
            return y_out, 0.0
        
        else:
            # --- Invasive (FXLMS) Path ---
            # 1. Generate anti-noise with current weights
            y_out = self.controller.generate_actuation(x_n)
            
            # 2. Get secondary path estimate and probe signal from engine
            s_hat, v_n = self.modeling_engine.update(x_n, e_n, y_out)
            
            # 3. Update controller weights using filtered-x LMS
            self.controller.solve_or_update(None, s_hat, x_n, e_n)
            
            # 4. Return total actuation (control + probe) and probe separately for logging
            return y_out + v_n, v_n