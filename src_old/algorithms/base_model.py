class SecondaryPathModel:
    def __init__(self, s_taps):
        self.s_hat = np.zeros(s_taps) # The digital twin of S

    def update(self, error_sample, ref_sample, secondary_output):
        """
        All models must take these three inputs to update self.s_hat.
        """
        raise NotImplementedError("Subclasses must implement update logic.")