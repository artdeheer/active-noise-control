"""
Rational sample-rate conversion for device fs ↔ process fs (e.g. 48 kHz ↔ 2.5 kHz).

Uses scipy.signal.resample_poly:  output_rate = input_rate * up / down.
"""

from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly


def poly_factors(fs_from: int, fs_to: int) -> tuple[int, int]:
    """(up, down) so that fs_from * up / down == fs_to (exact rational)."""
    fr = Fraction(fs_to, fs_from).limit_denominator(50000)
    return fr.numerator, fr.denominator


def device_frames_per_process_block(fs_device: int, fs_process: int) -> int:
    """Input length (device-rate samples) for one polyphase block to decimate cleanly."""
    _, down = poly_factors(fs_device, fs_process)
    return down


def resample_to_rate(x: np.ndarray, fs_from: int, fs_to: int) -> np.ndarray:
    """1-D float array from fs_from to fs_to."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if len(x) == 0:
        return x
    up, down = poly_factors(fs_from, fs_to)
    return resample_poly(x, up, down)
