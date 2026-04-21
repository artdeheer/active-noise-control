import numpy as np

import matplotlib
matplotlib.use('Agg') # This tells Matplotlib NOT to look for a window/display
import matplotlib.pyplot as plt

# 1. System Parameters
fs = 8000          # Sampling frequency (8kHz)
duration = 1.0     # 1 second
t = np.arange(0, duration, 1/fs)
n_samples = len(t)

# 2. Generate "Real World" Noise (a 100Hz tone)
# This is what the Reference Mic hears
ref_noise = 0.5 * np.sin(2 * np.pi * 100 * t)

# 3. Initialize ANC Parameters
mu = 0.01          # Step size (The "Learning Rate")
n_taps = 32        # Filter length (How many samples the CPU looks at once)
weights = np.zeros(n_taps)  # The digital filter weights
buffer = np.zeros(n_taps)   # Input buffer (Delay line)

# Output arrays for plotting
output_antinoise = np.zeros(n_samples)
error_signal = np.zeros(n_samples)

# 4. The Real-Time Loop (LMS Algorithm)
for i in range(n_samples):
    # Step A: Slide new sample into buffer
    buffer = np.roll(buffer, 1)
    buffer[0] = ref_noise[i]
    
    # Step B: Calculate Anti-Noise (The Filter Output)
    # y = sum(weights * buffer)
    antinoise = np.dot(weights, buffer)
    output_antinoise[i] = antinoise
    
    # Step C: Calculate the Error (What hits the Error Mic)
    # Ideally: Noise + Anti-noise = 0
    # In ANC math, antinoise is subtracted
    e = ref_noise[i] - antinoise 
    error_signal[i] = e
    
    # Step D: Update Weights (The "Learning" part)
    # weights = weights + 2 * mu * error * buffer
    weights += 2 * mu * e * buffer

# 5. Visualize the Results
plt.figure(figsize=(10, 6))
plt.plot(t[:500], ref_noise[:500], label="Original Noise", alpha=0.5)
plt.plot(t[:500], error_signal[:500], label="Result at Error Mic (Silenced)", color='red')
plt.title("ANC Virtual Model: Convergence of LMS Algorithm")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.grid(True)
plt.savefig("anc_results.png") # Save it to your folder
print("Simulation finished. Result saved as 'anc_results.png'")