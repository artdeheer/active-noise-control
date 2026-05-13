import os
import json
import sys
from pathlib import Path

import numpy as np
from datetime import datetime
import matplotlib

_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.append(str(_SRC))

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as cfg
from virtual_duct import VirtualDuct
from anc_brain import ANCBrain
from invasive.engine import InvasiveEngine 

def run_simulation(engine_type="invasive"):
    fs = cfg.FS  
    duration = 100  
    n_samples = fs * duration

    duct = VirtualDuct(cfg, use_acoustic_secondary_delay=False)
    engine = InvasiveEngine()
    brain = ANCBrain(engine)

    error_history, noise_history, anti_noise_history = [], [], []

    print(f"Starting {engine_type} simulation...")
    current_e_n = 0 

    for n in range(n_samples):
        t = n / fs
        # Your specific working signal
        x_n = 0.5 * np.sin(2 * np.pi * 100 * t) + 0.2 * np.sin(2 * np.pi * 400 * t)
        
        # 1. Process Sample (Logic Step)
        if n < 1000:
            s_hat, v_n = brain.modeling_engine.update(x_n, current_e_n, 0)
            y_out = v_n
        else:
            y_out = brain.process_sample(x_n, current_e_n)
        
        # 2. Physics Step
        d_n, current_e_n = duct.simulate_step(x_n, y_n=y_out)
        
        # 3. Storage
        error_history.append(current_e_n)
        noise_history.append(d_n)
        anti_noise_history.append(y_out)

    return noise_history, error_history, anti_noise_history

def save_run(engine_type, noise, error, anti):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("results", engine_type, "virtual", timestamp)
    os.makedirs(run_dir, exist_ok=True)

    # --- COMPLETE CONFIG LOGGING ---
    # This loop grabs every constant defined in your config.py
    complete_config = {}
    for item in dir(cfg):
        if item.isupper(): # Conventional way to grab constants
            value = getattr(cfg, item)
            # Convert numpy arrays/lists to strings/lists for JSON compatibility
            if isinstance(value, np.ndarray):
                complete_config[item] = value.tolist()
            else:
                complete_config[item] = value

    with open(os.path.join(run_dir, 'complete_config.json'), 'w') as f:
        json.dump(complete_config, f, indent=4)

    # --- Plots ---
    # 01 Performance
    plt.figure(figsize=(12, 6))
    plt.plot(noise, label='Noise', color='gray', alpha=0.5)
    plt.plot(error, label='Error', color='blue', linewidth=0.5)
    plt.title(f"Performance: {engine_type.upper()}")
    plt.savefig(os.path.join(run_dir, '01_performance.png'))
    plt.close()

    # 02 Steady State
    plt.figure(figsize=(10, 4))
    plt.plot(noise[-100:], label='Noise', color='gray')
    plt.plot(error[-100:], label='Error', color='blue')
    plt.title("Steady State Zoom")
    plt.savefig(os.path.join(run_dir, '02_zoom.png'))
    plt.close()

    # 03 Phase Check
    plt.figure(figsize=(10, 4))
    plt.plot(noise[-100:], label='Noise', color='gray', linestyle='--')
    plt.plot(anti[-100:], label='Anti-Noise', color='red')
    plt.title("Phase Check")
    plt.savefig(os.path.join(run_dir, '03_phase.png'))
    plt.close()

    print(f"Results and full config saved to: {run_dir}")

if __name__ == "__main__":
    algorithm = "invasive"
    n, e, a = run_simulation(algorithm)
    save_run(algorithm, n, e, a)
    print("Run complete.")