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

# Import Modeling Engines
from algorithms.invasive import InvasiveEngine 
from algorithms.non_invasive import NonInvasiveEngine

# Import Controllers
from controllers.invasive_controller import InvasiveController
from controllers.non_invasive_controller import NonInvasiveController

def run_simulation(engine_type="orthogonal"):
    fs = cfg.FS  
    duration = 30  
    n_samples = fs * duration
    duct = VirtualDuct()
    m_taps = cfg.S_TAPS 
    
    # Logic to respect the engine_type argument
    if engine_type.lower() == "orthogonal":
        engine = NonInvasiveEngine(m=m_taps)
        controller = NonInvasiveController(m=m_taps)
        is_ortho = True
    else:
        engine = InvasiveEngine() 
        controller = InvasiveController()
        is_ortho = False

    # Create the brain to coordinate everything
    brain = ANCBrain(engine, controller, is_orthogonal=is_ortho)

    history = {"e": [], "d": [], "a": [], "v": []}
    current_e_n = 0

    print(f"Starting {engine_type} simulation...")

    for n in range(n_samples):
        t = n / fs
        # Reference signal: sum of sinusoids
        x_n = 0.5 * np.sin(2 * np.pi * 100 * t) + 0.2 * np.sin(2 * np.pi * 400 * t)
        
        # Environment change handling
        if n == (n_samples // 2):
            print(f"\n[!] ALERT: Changing duct physics...")
            duct.change_environment() 

        # Process sample through brain (handles modeling, control, and probing)
        y_out, v_probe = brain.process_sample(x_n, current_e_n)
        
        # Physical Simulation
        d_n, current_e_n = duct.simulate_step(x_n, y_n=y_out)
        
        # Logging
        history["e"].append(current_e_n)
        history["d"].append(d_n)
        history["a"].append(y_out)
        history["v"].append(v_probe)

    return history

def save_run(engine_type, history, custom_name=None):
    # Determine directory name
    if custom_name:
        run_dir = os.path.join("results", engine_type, custom_name)
        # Always overwrite if it exists
        if os.path.exists(run_dir):
            print(f"Directory '{run_dir}' exists. Overwriting...")
            shutil.rmtree(run_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join("results", engine_type, timestamp)

    os.makedirs(run_dir, exist_ok=True)

    e, d, a = np.array(history["e"]), np.array(history["d"]), np.array(history["a"])

    # Calculate metrics
    mse = np.mean(e**2)
    reduction_db = 10 * np.log10(np.mean(d[:1000]**2) / (np.mean(e[-1000:]**2) + 1e-10))

    scores = {"mse_total": float(mse), "reduction_db": float(reduction_db)}

    with open(os.path.join(run_dir, 'scores.json'), 'w') as f:
        json.dump(scores, f, indent=4)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(d, label='Primary Noise (d_n)', color='gray', alpha=0.5)
    plt.plot(e, label='Error Signal (e_n)', color='blue', linewidth=0.5)
    plt.title(f"Performance: {engine_type.upper()} {'('+custom_name+')' if custom_name else ''}")
    plt.legend()
    plt.savefig(os.path.join(run_dir, '01_performance.png'))
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(d[-100:], label='Primary Noise', color='gray')
    plt.plot(e[-100:], label='Residual Error', color='blue')
    plt.title("Steady State Zoom")
    plt.legend()
    plt.savefig(os.path.join(run_dir, '02_zoom.png'))
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(d[-100:], label='Target', color='gray', linestyle='--')
    plt.plot(a[-100:], label='Anti-Noise', color='red')
    plt.title("Phase Check")
    plt.legend()
    plt.savefig(os.path.join(run_dir, '03_phase.png'))
    plt.close()

    print(f"Successfully saved run to: {run_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ANC Simulation with Engine Selection")
    
    # Algorithm selection
    parser.add_argument("algorithm", 
                        choices=["invasive", "orthogonal"], 
                        nargs='?', 
                        default="orthogonal",
                        help="Engine type: 'invasive' or 'orthogonal' (default: %(default)s)")
    
    # Custom name flag
    parser.add_argument("--name", "-n", 
                        type=str, 
                        help="Optional custom name for the results folder. Overwrites existing folders.")

    args = parser.parse_args()

    # Execution
    hist = run_simulation(args.algorithm)
    save_run(args.algorithm, hist, custom_name=args.name)