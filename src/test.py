import numpy as np
import time
from io_manager import IOManager
from non_invasive.engine import NonInvasiveEngine
from non_invasive.controller import NonInvasiveController

# Quick full-rate loop (device fs only). For logging + 2.5 kHz decimation use:
#   cd src && uv run python run_physical_orthogonal.py --duration 30

def run_test():
    # 1. Setup Parameters (Section V: Experimental Verification)
    # Paper uses 2.5 kHz [cite: 283]; USB interfaces need standard rates (48k / 44.1k).
    M = 64          # Number of taps (reduced for 1m pipe) [cite: 263, 303]

    # 2. Initialize Components (actual rate is io.fs after PortAudio opens the device)
    io = None
    io = IOManager(fs=48000, chunk=1)
    FS = io.fs
    engine = NonInvasiveEngine(m=M)
    controller = NonInvasiveController(m=M, update_alpha=0.1) # Smooth updates

    print("--- ANC test (Behringer UMC204HD) ---")
    print("I/O: mic → Input 1 (reference), mic → Input 2 (error); anti-noise → Main L + Main R")
    print(f"Sampling rate: {FS} Hz | FIR taps: {M}")
    print("Run from the src/ directory (e.g. cd src && python test.py). Ctrl+C to stop.")

    last_a_n = 0.0
    start_time = time.time()

    try:
        while True:
            # 1. Capture r_n (Ref Mic) and e_n (Error Mic)
            # The IOManager sends last_a_n to the speaker simultaneously
            r_n, e_n = io.capture_and_play(last_a_n)

            # 2. Modeling Step (Equation 12)
            # Simultaneous identification of H and S paths [cite: 103, 164]
            h_hat, s_hat = engine.update(r_n, last_a_n, e_n)

            # 3. Controller Update (Section IV.B: FIR Solution)
            # Only update the controller periodically to save CPU [cite: 264, 304]
            if int(time.time() * 10) % 2 == 0: # Update ~5 times per second
                controller.solve_or_update(h_hat, s_hat)

            # 4. Generate Anti-Noise (Equation 1)
            # Noninvasive: No probing signals used [cite: 9, 354]
            a_n = controller.generate_actuation(r_n)
            last_a_n = a_n

            # 5. Telemetry (Optional: Print status every second)
            if time.time() - start_time > 1.0:
                s_norm = np.linalg.norm(s_hat)
                print(f"Status | Error Mic Amp: {abs(e_n):.4f} | S-Path Model Norm: {s_norm:.4f}")
                start_time = time.time()

    except KeyboardInterrupt:
        print("\nStopping system...")
    finally:
        if io is not None:
            io.close()

if __name__ == "__main__":
    run_test()