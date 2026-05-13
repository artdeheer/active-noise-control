"""
Discrete-time duct twin for offline simulation.

Uses any module that exposes PRIMARY_DELAY_SAMPLES and SECONDARY_DELAY_SAMPLES
(see invasive/config.py or non_invasive/config.py).
"""

import numpy as np


class VirtualDuct:
    def __init__(
        self,
        cfg,
        filter_len=10,
        use_acoustic_secondary_delay=True,
    ):
        """
        use_acoustic_secondary_delay:
            True  — secondary contribution uses samples delayed by
                    SECONDARY_DELAY_SAMPLES (matches the non-invasive / duct
                    experiment layout).
            False — legacy invasive plant: FIR acts on the most recent L
                    anti-noise samples only (old invasive/virtual_duct.py).
                    The FxLMS loop in invasive/ was tuned for this simpler plant.
        """
        self.cfg = cfg
        self.use_acoustic_secondary_delay = use_acoustic_secondary_delay
        self.buffer_P = np.zeros(cfg.PRIMARY_DELAY_SAMPLES)
        self.filter_len = filter_len
        if use_acoustic_secondary_delay:
            self.buffer_S = np.zeros(self.filter_len + cfg.SECONDARY_DELAY_SAMPLES)
        else:
            self.buffer_S = np.zeros(
                max(self.filter_len, cfg.SECONDARY_DELAY_SAMPLES)
            )
        self.L = self.filter_len
        self.generate_s_truth(damping=0.1, freq=0.1, gain=1.0)

    def simulate_step(self, x_n, y_n):
        d_n_pure = self.buffer_P[-1]
        d_n = d_n_pure * 0.95

        self.buffer_P[1:] = self.buffer_P[:-1]
        self.buffer_P[0] = x_n

        self.buffer_S[1:] = self.buffer_S[:-1]
        self.buffer_S[0] = y_n

        if self.use_acoustic_secondary_delay:
            delay = self.cfg.SECONDARY_DELAY_SAMPLES
            arrival_window = self.buffer_S[delay : delay + self.L]
        else:
            arrival_window = self.buffer_S[: self.L]
        y_prime = np.dot(arrival_window, self.s_truth)
        e_n = d_n + y_prime

        return d_n, e_n

    def generate_s_truth(self, damping, freq, gain=1.0):
        n_axis = np.arange(self.L)
        decay_envelope = np.exp(-damping * n_axis)
        oscillation = np.sin(2 * np.pi * freq * n_axis)
        self.s_truth = gain * (decay_envelope * oscillation)

    def change_environment(self, damping=0.05, freq=0.15, gain=-1.0):
        print(f"--- Physical Environment Changed: freq={freq}, gain={gain} ---")
        self.generate_s_truth(damping, freq, gain)
