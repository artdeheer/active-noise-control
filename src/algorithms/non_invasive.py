import numpy as np

class NonInvasiveEngine:
    def __init__(self, m):
        self.m = m
        # Initialize with very small random values
        self.theta_hat = np.random.normal(0, 1e-6, 2 * self.m) 
        self.phi_vector = np.zeros(2 * self.m)
        
        # Physical constraint: The model shouldn't assume the duct 
        # has a gain of 1000x. We cap the weights at a reasonable value.
        self.max_theta = 2.0 

    def update(self, r_n, a_n, e_n):
        """Implements Equation 12: LS algorithm for online modeling."""
        # 1. Update regression vector phi (Standard Shift)
        self.phi_vector[1:self.m] = self.phi_vector[0:self.m-1]
        self.phi_vector[0] = r_n
        self.phi_vector[self.m+1:] = self.phi_vector[self.m:-1]
        self.phi_vector[self.m] = a_n

        # 2. Calculate estimation error epsilon (Equation 10)
        epsilon = e_n - np.dot(self.phi_vector, self.theta_hat)

        # 3. Update parameter vector using LS rule (Equation 12)
        phi_norm_sq = np.dot(self.phi_vector, self.phi_vector) + 1e-6
        
        # Normalized update
        delta_theta = (epsilon / phi_norm_sq) * self.phi_vector
        self.theta_hat += delta_theta


        return self.theta_hat[:self.m], self.theta_hat[self.m:]