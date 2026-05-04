import numpy as np
import matplotlib
# Force Matplotlib to use a non-interactive backend (Agg)
# This prevents the '_tkinter.TclError: no display name' error
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

from virtual_duct import VirtualDuct
from anc_brain import ANCBrain
from algorithms.non_invasive import NonInvasiveEngine
from algorithms.invasive import InvasiveEngine # Import both for comparison

def run_simulation(engine_type="non_invasive"):
    # 1. Setup Environment
    fs = 2500  
    duration = 5  
    n_samples = fs * duration

    # Initialize components
    duct = VirtualDuct()
    
    if engine_type == "invasive":
        engine = InvasiveEngine()
    else:
        engine = NonInvasiveEngine()
        
    brain = ANCBrain(engine)

    # 2. Storage
    error_history = []
    noise_history = []

    print(f"Starting {engine_type} simulation...")

    # 3. THE MAIN LOOP
    current_e_n = 0 
    for n in range(n_samples):
        t = n / fs
        x_n = 0.5 * np.sin(2 * np.pi * 100 * t) + 0.2 * np.sin(2 * np.pi * 400 * t)
        
        if n < 500:
            # --- WARM UP PHASE ---
            # We don't want the brain to try to cancel yet (y_n = 0)
            # But we DO want the modeling engine to learn the path!
            s_hat, v_n = brain.modeling_engine.update(x_n, current_e_n, 0)
            y_total = v_n # Only play the white noise hiss
        else:
            # --- ACTIVE PHASE ---
            y_total = brain.process_sample(x_n, current_e_n)
        
        # Step 2: Duct simulates physics
        d_n, current_e_n = duct.simulate_step(x_n, y_n=y_total)
        
        # Save results
        error_history.append(current_e_n)
        noise_history.append(d_n)

        # Updated debug block in main.py
        if n == 1000:
            # We check the weights; if they aren't 0.0, the brain is learning!
            weight_sum = np.sum(np.abs(brain.h_weights))
            print(f"Sample {n}: Brain Activity (Sum of Absolute Weights) = {weight_sum}")

    return noise_history, error_history

# --- RUN EXPERIMENTS ---
# Run Non-Invasive
noise_ni, error_ni = run_simulation("invasive")

# Run Invasive (Optional: Uncomment after you create invasive.py)
# noise_inv, error_inv = run_simulation("invasive")

# --- VISUALIZATION ---
plt.figure(figsize=(12, 6))

plt.plot(noise_ni, label='Primary Noise (Original)', color='gray')
plt.plot(error_ni, label='Residual Error', color='blue', linewidth=1)


plt.title("Active Noise Control Performance: Primary Noise vs Residual Error")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.grid(True, alpha=1)
plt.legend(loc='upper right') # Forces legend to a clear spot

# Save the file
output_file = 'anc_results.png'
plt.savefig(output_file, dpi=300)
print(f"Simulation complete. Plot saved as {output_file}")