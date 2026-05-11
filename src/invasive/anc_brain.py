import numpy as np
import config as cfg

class ANCBrain:
    def __init__(self, modeling_engine):
        # The Adaptive Filter (H weights)
        # This is the brain's internal 'counter-strategy'
        self.h_weights = np.zeros(cfg.M_TAPS) # the amount of weights has to be equal to the buffer length of incoming noise for the calculations later
        self.x_buffer = np.zeros(cfg.M_TAPS) # the length of the buffer should be equal to the time it takes the noise to travel
        
        self.modeling_engine = modeling_engine # the modeling engine, should either be removed or combined into one file with non invasive brain

    def process_sample(self, x_n, e_n):
        # x_n is incoming noise, this slides the kept buffer
        self.x_buffer[1:] = self.x_buffer[:-1]
        self.x_buffer[0] = x_n

        y_n = np.dot(self.x_buffer, self.h_weights) # the anti noise (2)

        # Get the latest path guess (s_hat) from the engine
        # For Invasive, this also gives us the white noise v_n
        s_hat, v_n = self.modeling_engine.update(x_n, e_n, y_n)

        # filtered x, filter x through the secondary path (s_hat) to predict how the noise will sound at the error mic
        # doing dot product with the last noise in the buffer of x, multiplied by s_hat
        x_filtered = np.dot(self.x_buffer[:len(s_hat)], s_hat) # (3)


        reg = 1e-6  # small number to prevent division by zero
        x_pow = np.dot(x_filtered, x_filtered) + reg # calculates the energy currently in flight between speaker and microphone

        # Normalized update
        # Purely multiplication-based update (Standard LMS)
        # self.h_weights -= cfg.MU_CONTROLLER * e_n * x_filtered # (LMS)
        self.h_weights -= (cfg.MU_CONTROLLER / x_pow) * e_n * x_filtered # update the weights (NLMS)
        # added division such that a loud bang or something would not make the algorithm explode 
        # cause if noise gets 10x bigger, the the update also gets 10x bigger

        return y_n + v_n # even if the antinoise is "perfect", the noise will never be truly silent since white noise is added