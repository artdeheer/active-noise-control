class ANCBrain:
    def __init__(self, modeling_engine, controller):
        self.modeling_engine = modeling_engine
        self.controller = controller

    def generate_actuation(self, r_n):
        """
        Generate the current actuation from the latest FIR controller.
        Returns: (y_out, v_probe) for interface compatibility with the invasive path.
        """
        y_out = self.controller.generate_actuation(r_n)
        return y_out, 0.0

    def adapt(self, r_n, a_n, e_n):
        """
        Update the path models with samples from the same time index and
        refresh the FIR controller for the next sample.
        """
        h_hat, s_hat = self.modeling_engine.update(r_n, a_n, e_n)
        self.controller.solve_or_update(h_hat, s_hat)
