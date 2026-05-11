from scipy.linalg import toeplitz
import numpy as np


class NonInvasiveController:
    def __init__(
        self,
        m,
        startup_samples=0,
        startup_actuation_limit=None,
        update_alpha=1.0,
        max_actuation_step=None,
    ):
        self.m = m
        self.g_weights = np.zeros(self.m)
        self.r_buffer = np.zeros(self.m)
        self.startup_samples = startup_samples
        self.startup_actuation_limit = startup_actuation_limit
        self.update_alpha = update_alpha
        self.max_actuation_step = max_actuation_step
        self.generated_samples = 0
        self.last_actuation = 0.0

    def solve_or_update(self, h_hat, s_hat):
        """Solve the direct FIR normal equation R_s g = -p."""
        h_hat = np.asarray(h_hat, dtype=float)
        s_hat = np.asarray(s_hat, dtype=float)

        if np.linalg.norm(s_hat) < 1e-12:
            return

        # Equation (29): Toeplitz autocorrelation matrix of the secondary path.
        r_elements = np.array(
            [np.dot(s_hat[: self.m - k], s_hat[k:]) for k in range(self.m)],
            dtype=float,
        )
        Rs = toeplitz(r_elements)

        # Equation (28): p = M_s^T h_hat.
        p = np.array(
            [np.dot(s_hat[: self.m - k], h_hat[k:]) for k in range(self.m)],
            dtype=float,
        )

        try:
            target_weights = np.linalg.solve(Rs, -p)
        except np.linalg.LinAlgError:
            target_weights = -np.linalg.pinv(Rs) @ p

        self.g_weights = (
            (1.0 - self.update_alpha) * self.g_weights
            + self.update_alpha * target_weights
        )

    def generate_actuation(self, r_n):
        self.r_buffer[1:] = self.r_buffer[:-1]
        self.r_buffer[0] = r_n

        actuation = float(np.dot(self.r_buffer, self.g_weights))

        # Protect startup while the secondary-path estimate is still bootstrapping.
        if (
            self.startup_actuation_limit is not None
            and self.generated_samples < self.startup_samples
        ):
            actuation = float(
                np.clip(
                    actuation,
                    -self.startup_actuation_limit,
                    self.startup_actuation_limit,
                )
            )

        # Keep the controller output continuous when the FIR weights change abruptly.
        if self.max_actuation_step is not None:
            actuation = float(
                np.clip(
                    actuation,
                    self.last_actuation - self.max_actuation_step,
                    self.last_actuation + self.max_actuation_step,
                )
            )

        self.last_actuation = actuation
        self.generated_samples += 1
        return actuation
