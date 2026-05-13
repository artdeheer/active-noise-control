"""
Real-acoustics path: microphones + loudspeakers via the audio interface.

Simulation uses VirtualDuct.simulate_step(x_n, y_n) -> (d_n, e_n).
Hardware does not give you d_n unless you measure it; the ANC loop uses
reference r_n and error e_n from IOManager (Input 1 / Input 2).

For a full live loop with optional decimation to 2.5 kHz (matching
``non_invasive/config.py`` FS) and saved plots, use::

  cd src && uv run python run_physical_orthogonal.py --duration 30

Invasive hardware (includes probe noise)::

  cd src && uv run python run_physical_invasive.py --duration 20
"""

from io_manager import IOManager


class PhysicalDuct:
    """Thin wrapper so the live path stays in one place beside VirtualDuct."""

    def __init__(self, io=None, **io_kwargs):
        self.io = io if io is not None else IOManager(**io_kwargs)

    def step(self, actuation):
        """One full-duplex sample: play actuation, return (r_n, e_n)."""
        return self.io.capture_and_play(actuation)

    def close(self):
        self.io.close()
