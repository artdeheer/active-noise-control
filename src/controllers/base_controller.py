from abc import ABC, abstractmethod
import numpy as np

class BaseController(ABC):
    @abstractmethod
    def solve_or_update(self, h_hat, s_hat, x_n, e_n):
        """Update weights (FXLMS) or solve (Orthogonal)."""
        pass

    @abstractmethod
    def generate_actuation(self, r_n):
        """Produce the signal for the speaker."""
        pass