import os
import json
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import config as cfg
from virtual_duct import VirtualDuct
from anc_brain import ANCBrain
from algorithms.invasive import InvasiveEngine 

def run_simulation(engine_type="invasive"):
    fs = cfg.FS  
    duration = 100  
    n_samples = fs * duration

    duct = VirtualDuct()
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
    run_dir = os.path.join("results", engine_type, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    # Log parameters
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump({
            "MU_CONTROLLER": cfg.MU_CONTROLLER,
            "MU_MODELING": cfg.MU_MODELING,
            "M_TAPS": cfg.M_TAPS
        }, f, indent=4)

    # Plots
    plt.figure(figsize=(12, 6))
    plt.plot(noise, label='Noise', color='gray', alpha=0.5)
    plt.plot(error, label='Error', color='blue', linewidth=0.5)
    plt.title(f"Performance: {engine_type.upper()}")
    plt.savefig(os.path.join(run_dir, '01_performance.png'))
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(noise[-100:], label='Noise', color='gray')
    plt.plot(error[-100:], label='Error', color='blue')
    plt.title("Steady State")
    plt.savefig(os.path.join(run_dir, '02_zoom.png'))
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(noise[-100:], label='Noise', color='gray', linestyle='--')
    plt.plot(anti[-100:], label='Anti-Noise', color='red')
    plt.title("Phase Check")
    plt.savefig(os.path.join(run_dir, '03_phase.png'))
    plt.close()

if __name__ == "__main__":
    n, e, a = run_simulation("invasive")
    save_run("invasive", n, e, a)
    print("Run complete.")