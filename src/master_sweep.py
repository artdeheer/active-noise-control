import os
import csv
import numpy as np
import config as cfg
from virtual_duct import VirtualDuct
from anc_brain import ANCBrain

# Adjust these imports based on your exact file structure
from algorithms.invasive import InvasiveEngine
from controllers.invasive_controller import InvasiveController

def run_static_trial(m_mu, c_mu, taps):
    fs = cfg.FS
    # Longer duration (15s) helps verify if slow-learning settings eventually win
    duration = 15  
    n_samples = fs * duration
    duct = VirtualDuct()
    
    # Force the sweep parameters into the config
    cfg.MU_MODELING = float(m_mu)
    cfg.MU_CONTROLLER = float(c_mu)
    cfg.M_TAPS = int(taps) 
    
    engine = InvasiveEngine()
    controller = InvasiveController()
    brain = ANCBrain(engine, controller, is_orthogonal=False)

    # We only measure MSE in the last 5 seconds to ignore the initial startup/probing
    steady_state_errors = []
    measurement_start = fs * 10 
    e_n = 0.0

    for n in range(n_samples):
        t = n / fs
        x_n = 0.5 * np.sin(2 * np.pi * 100 * t) + 0.2 * np.sin(2 * np.pi * 400 * t)
        
        # ENVIRONMENT REMAINS STATIC
        # (No duct.change_environment() calls here)
        
        # Process through brain
        y_out, v_probe = brain.process_sample(x_n, e_n)
        
        # Physical simulation
        d_n, e_n = duct.simulate_step(x_n, y_n=y_out)
            
        if np.isnan(e_n) or np.abs(e_n) > 50:
            return 999.0
            
        if n > measurement_start:
            steady_state_errors.append(e_n**2)

    return np.mean(steady_state_errors)

def master_static_sweep():
    # 1. SEARCH SPACE
    # Modeling doesn't need to be super fast in a static environment
    mu_m_grid = np.logspace(-4, -1, num=8)   
    mu_c_grid = np.logspace(-6, -3, num=10)  
    taps_grid = [64, 128, 256, 512]          
    
    trials_per_config = 3 
    
    results = []
    sweep_dir = "results/static_sweep"
    os.makedirs(sweep_dir, exist_ok=True)

    total_configs = len(mu_m_grid) * len(mu_c_grid) * len(taps_grid)
    print(f"🔬 STARTING STATIC STEADY-STATE SWEEP")
    print(f"Configurations: {total_configs}")
    print("="*75)

    count = 0
    for taps in taps_grid:
        for m_mu in mu_m_grid:
            for c_mu in mu_c_grid:
                count += 1
                trial_results = []
                
                for t in range(trials_per_config):
                    mse = run_static_trial(m_mu, c_mu, taps)
                    trial_results.append(mse)

                median_mse = np.median(trial_results)
                # Lower threshold for pass because we expect better results in static env
                is_stable = median_mse < 0.3 

                results.append({
                    "Taps": taps,
                    "Mu_M": m_mu,
                    "Mu_C": c_mu,
                    "Steady_State_MSE": median_mse,
                    "Stability": "PASS" if is_stable else "FAIL"
                })

                status = "✅" if is_stable else "💥"
                if count % 10 == 0 or is_stable:
                    print(f"[{count:03d}/{total_configs}] Taps:{taps:<3} M:{m_mu:.1e} C:{c_mu:.1e} | MSE:{median_mse:.5f} | {status}")

    # 2. RANKING
    stable_results = [r for r in results if r["Stability"] == "PASS"]
    hall_of_fame = sorted(stable_results, key=lambda x: x["Steady_State_MSE"])

    print("\n" + "🏆" * 25)
    print("  STATIC ENVIRONMENT CHAMPIONS  ")
    print("🏆" * 25)
    if not hall_of_fame:
        print("NO STABLE CONFIGS FOUND.")
    else:
        print(f"{'Rank':<5} | {'Taps':<5} | {'Mu_M':<10} | {'Mu_C':<10} | {'Steady-MSE':<10}")
        print("-" * 65)
        for rank, res in enumerate(hall_of_fame[:15]):
            print(f"#{rank+1:<4} | {res['Taps']:<5} | {res['Mu_M']:<10.1e} | {res['Mu_C']:<10.1e} | {res['Steady_State_MSE']:<10.6f}")

    # 3. SAVE
    csv_path = os.path.join(sweep_dir, "static_report.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

if __name__ == "__main__":
    master_static_sweep()