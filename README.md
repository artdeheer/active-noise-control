# ANC Online Secondary Path Modeling: Comparative Analysis

## Project Overview
This repository contains a Python-based Active Noise Control (ANC) system designed for a 2-meter PVC duct (110 mm diameter). The project investigates the trade-offs between invasive white noise injection and non-invasive orthogonal adaptation for online secondary path modeling, focusing on a 100 Hz–400 Hz frequency range.

## File Tree
.
├── main.py                 # Entry point and real-time loop coordinator
├── config.py               # Physical constants and system hyperparameters
├── io_manager.py           # Hardware/Simulation I/O abstraction layer
├── virtual_duct.py         # Digital twin of the physical acoustic environment
├── anc_brain.py            # Main controller managing buffers and FxLMS
├── analysis.py             # Performance metrics and data visualization
├── algorithms/             # Modeling strategy implementations
│   ├── base_model.py       # Abstract base class for modeling engines
│   ├── invasive.py         # White noise injection modeling logic
│   └── non_invasive.py     # Yuan’s orthogonal adaptation math
└── README.md               # Project documentation

## Component Breakdown

### 1. Global Configuration (config.py)
This file serves as the single source of truth for the project. It defines:
*   **Physical Constants**: Speed of sound (343 m/s) and duct dimensions.
*   **DSP Specs**: Sampling frequency (2500 Hz), target tones (100–400 Hz), and filter tap lengths.
*   **Hyperparameters**: Learning rates (mu) for both the controller and modeling engines.

### 2. The Acoustic Environment (virtual_duct.py)
While the physical setup is pending, this module simulates the 2-meter PVC pipe:
*   **Primary Path**: Replicates the acoustic delay from the noise source to the error microphone.
*   **Secondary Path**: Simulates the transfer function of the secondary speaker and microphone using a fixed "ground truth" FIR filter.
*   **Summation**: Performs the acoustic addition of primary noise and anti-noise.

### 3. The Controller (anc_brain.py)
The central processing unit that handles the real-time logic:
*   **Buffer Management**: Maintains sliding windows for reference noise and filtered-x signals.
*   **Control Loop**: Executes the Filtered-x Least Mean Square (FxLMS) update.
*   **Strategy Orchestration**: Delegates path modeling to the selected engine in the `algorithms/` directory.

### 4. Path Modeling Engines (algorithms/)
This directory contains the logic for the two methods being compared:
*   **Invasive (invasive.py)**: Identifies the secondary path by injecting a low-level broadband white noise signal and correlating it with the error mic input.
*   **Non-Invasive (non_invasive.py)**: Implements Yuan’s Orthogonal Adaptation. It extracts the secondary path model from the existing noise signals without auxiliary excitation.

### 5. Evaluation & Analysis (analysis.py)
Used to generate data for the thesis results:
*   **Acoustic Metrics**: Measures steady-state attenuation (dB) and convergence speed.
*   **Computational Metrics**: Benchmarks CPU utilization and execution time for each algorithm to evaluate feasibility on general-purpose hardware.

## Workflow
The system processes data sample-by-sample. `main.py` fetches inputs via `io_manager.py`, passes them to `anc_brain.py` for processing, and outputs the resulting anti-noise. Performance data is logged in real-time and processed by `analysis.py` upon termination.