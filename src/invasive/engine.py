import numpy as np
import config as cfg

class InvasiveEngine:
    def __init__(self):
        filter_len = 10
        # 1. The 'Guess' of the secondary path
        self.s_hat = np.zeros(filter_len)
        
        # 2. Buffer for the SECRET probe signal (White Noise)
        self.v_buffer = np.zeros(filter_len)

    def update(self, x_n, e_n, y_n):
        """
        Engine A: Updates s_hat by injecting white noise.
        """
        # create a tiny bit of white noise
        v_n = np.random.normal(0, 0.01)

        # keep the history of white noise
        self.v_buffer[1:] = self.v_buffer[:-1]
        self.v_buffer[0] = v_n

        # calculate modeling error, 
        e_modeling = e_n - np.dot(self.v_buffer, self.s_hat)

        # update S hat
        self.s_hat += cfg.MU_MODELING * e_modeling * self.v_buffer

        # return s_hat and v_n so it can be added to the speaker output
        return self.s_hat, v_n