import os
import json
import sys
import shutil
import argparse
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# Internal Project Imports
import config as cfg
from virtual_duct import VirtualDuct
from anc_brain import ANCBrain

# Import Modeling Engine and Controller (Orthogonal/Non-Invasive focus)
from non_invasive.engine import NonInvasiveEngine
from non_invasive.controller import NonInvasiveController

def run_simulation():
    fs = cfg.FS  
    duration = 30  
    n_samples = fs * duration
    duct = VirtualDuct()
    m_taps = cfg.S_TAPS 
    
    # Initialize the Orthogonal components
    # Orthogonal method is non-invasive: it uses actual anti-noise to model the path
    engine = NonInvasiveEngine(m=m_taps)
    controller = NonInvasiveController(m=m_taps)

    # The brain coordinates the NonInvasiveEngine and NonInvasiveController
    brain = ANCBrain(engine, controller)

    history = {"e": [], "d": [], "a": [], "v": []}
    current_e_n = 0

    print("Starting ORTHOGONAL (Non-Invasive) simulation...")

    for n in range(n_samples):
        t = n / fs
        # Reference signal (the noise we want to cancel)
        x_n = 0.5 * np.sin(2 * np.pi * 100 * t) + 0.2 * np.sin(2 * np.pi * 400 * t)
        
        # Environment change handling (Halfway through)
        if n == (n_samples // 2):
            print(f"\n[!] ALERT: Physical environment shift...")
            duct.change_environment() 

        # Step 1: Generate the current control signal from the latest FIR weights.
        y_out, v_probe = brain.generate_actuation(x_n)
        
        # Step 2: Physics simulation (Duct)
        # Note: In the orthogonal method, total actuation is y_out + v_probe
        d_n, current_e_n = duct.simulate_step(x_n, y_n=(y_out + v_probe))

        # Step 3: Adapt with the reference, actuation, and error from the same sample.
        brain.adapt(x_n, y_out + v_probe, current_e_n)
        
        # Step 4: Logging
        history["e"].append(current_e_n)
        history["d"].append(d_n)
        history["a"].append(y_out)
        history["v"].append(v_probe)

    return history

def save_run(history, custom_name=None):
    # Determine directory name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = custom_name if custom_name else timestamp
    run_dir = os.path.join("results", "orthogonal", folder_name)

    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(run_dir, exist_ok=True)

    e, d, a = np.array(history["e"]), np.array(history["d"]), np.array(history["a"])

    # Calculate metrics
    mse = np.mean(e**2)
    # dB Reduction: Comparing initial noise to final steady-state error
    reduction_db = 10 * np.log10(np.mean(d[:1000]**2) / (np.mean(e[-1000:]**2) + 1e-10))

    scores = {"mse_total": float(mse), "reduction_db": float(reduction_db)}

    with open(os.path.join(run_dir, 'scores.json'), 'w') as f:
        json.dump(scores, f, indent=4)

    # Visualizing the performance
    plt.figure(figsize=(12, 6))
    plt.plot(d, label='Primary Noise (d_n)', color='gray', alpha=0.5)
    plt.plot(e, label='Error Signal (e_n)', color='blue', linewidth=0.5)
    plt.title(f"Performance: ORTHOGONAL {'('+custom_name+')' if custom_name else ''}")
    plt.legend()
    plt.savefig(os.path.join(run_dir, '01_performance.png'))
    plt.close()

    print(f"Simulation finished. Results saved to: {run_dir}")
    print(f"Steady-state Reduction: {reduction_db:.2f} dB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Orthogonal ANC Simulation")
    parser.add_argument("--name", "-n", type=str, help="Custom name for the result folder")
    args = parser.parse_args()

    # Run the cleaned-up Orthogonal simulation
    hist = run_simulation()
    save_run(hist, custom_name=args.name)
