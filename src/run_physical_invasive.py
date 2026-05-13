"""
Live invasive (FxLMS + probe) ANC on hardware — **loud** probing during startup.

Same multirate pattern as ``run_physical_orthogonal.py`` (default: 48 kHz ↔ 2.5 kHz).

Run from ``src/``::

  cd src && uv run python run_physical_invasive.py --duration 20

Results: ``results/invasive/physical/<timestamp>/``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "invasive"))

from anc_brain import ANCBrain
from engine import InvasiveEngine
from io_manager import IOManager
from multirate import device_frames_per_process_block, poly_factors, resample_to_rate


def _load_config(path: Path):
    spec = importlib.util.spec_from_file_location("inv_physical_cfg", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def save_invasive_physical(history, run_dir, meta):
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    e = np.asarray(history["e"])
    r = np.asarray(history["r"])
    y = np.asarray(history["y"])
    scores = {
        "mse_total": float(np.mean(e**2)),
        "reduction_db": float(
            10
            * np.log10(
                (np.mean(e[: min(500, len(e))] ** 2) + 1e-12)
                / (np.mean(e[-min(500, len(e)) :] ** 2) + 1e-12)
            )
        ),
    }
    with open(os.path.join(run_dir, "scores.json"), "w") as f:
        json.dump(scores, f, indent=2)
    plt.figure(figsize=(12, 6))
    plt.plot(r, label="Reference r_n", color="gray", alpha=0.5, linewidth=0.5)
    plt.plot(e, label="Error e_n", color="blue", linewidth=0.5)
    plt.title("Performance: INVASIVE (physical)")
    plt.legend()
    plt.savefig(os.path.join(run_dir, "01_performance.png"))
    plt.close()
    plt.figure(figsize=(10, 4))
    plt.plot(r[-500:], label="r_n", color="gray", alpha=0.6)
    plt.plot(y[-500:], label="Actuation y_n", color="red", linewidth=0.5)
    plt.title("Steady-state: reference vs actuation (last 500 control samples)")
    plt.legend()
    plt.savefig(os.path.join(run_dir, "02_actuation.png"))
    plt.close()
    print(f"Saved physical run to {run_dir}")


def main():
    parser = argparse.ArgumentParser(description="Invasive ANC on hardware (probe noise).")
    parser.add_argument("--duration", type=float, default=20.0, help="Seconds.")
    parser.add_argument(
        "--no-decimate",
        action="store_true",
        help="Run the controller at the device sample rate.",
    )
    parser.add_argument("--name", "-n", type=str, default=None, help="Result subfolder name.")
    args = parser.parse_args()

    inv_cfg = _load_config(SRC / "invasive" / "config.py")
    target_fs = int(inv_cfg.FS)

    io = IOManager(fs=48000, chunk=1)
    device_fs = int(io.fs)
    decimate = not args.no_decimate
    if decimate:
        blk = device_frames_per_process_block(device_fs, target_fs)
        up, dn = poly_factors(device_fs, target_fs)
        approx_fs = device_fs * up / dn
        if abs(approx_fs - target_fs) > 1e-3:
            io.close()
            raise SystemExit(
                f"No polyphase ratio between device_fs={device_fs} Hz and invasive/config.FS={target_fs} Hz "
                f"(would be {approx_fs:.4f} Hz). Use --no-decimate or 48000 Hz."
            )
        if blk != 1:
            io.close()
            io = IOManager(fs=device_fs, chunk=blk)
        control_fs = target_fs
    else:
        blk = 1
        control_fs = device_fs

    engine = InvasiveEngine()
    brain = ANCBrain(engine)

    n_blocks = max(1, int(args.duration * device_fs / blk))
    hist = {"e": [], "r": [], "y": []}
    out_high = np.zeros(blk, dtype=np.float32)
    global_low = 0

    print(
        f"Invasive physical | device_fs={device_fs} Hz | block={blk} | "
        f"decimate={'on' if decimate else 'off'} | control_fs≈{control_fs} Hz | "
        f"blocks≈{n_blocks} (Ctrl+C stops)"
    )
    t0 = time.time()
    try:
        for _ in range(n_blocks):
            r_hi, e_hi = io.exchange_block(out_high)
            if decimate:
                r_lo = resample_to_rate(r_hi, device_fs, inv_cfg.FS)
                e_lo = resample_to_rate(e_hi, device_fs, inv_cfg.FS)
            else:
                r_lo, e_lo = r_hi.astype(float), e_hi.astype(float)
            y_block = []
            for i in range(len(r_lo)):
                r_n, e_n = float(r_lo[i]), float(e_lo[i])
                if global_low < 1000:
                    _, v_n = engine.update(r_n, e_n, 0.0)
                    y_out = v_n
                else:
                    y_out = brain.process_sample(r_n, e_n)
                hist["r"].append(r_n)
                hist["e"].append(e_n)
                hist["y"].append(y_out)
                y_block.append(y_out)
                global_low += 1
            if decimate:
                out_high = resample_to_rate(
                    np.asarray(y_block, dtype=np.float64), inv_cfg.FS, device_fs
                ).astype(np.float32)
            else:
                out_high = np.asarray(y_block, dtype=np.float32)
    except KeyboardInterrupt:
        print("\nStopped early.")
    finally:
        io.close()

    if not hist["e"]:
        print("No samples captured; not writing results.")
        return

    elapsed = time.time() - t0
    folder = args.name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("results", "invasive", "physical", folder)
    meta = {
        "algorithm": "invasive",
        "mode": "physical",
        "device_fs_hz": device_fs,
        "process_fs_hz": inv_cfg.FS if decimate else device_fs,
        "decimated": decimate,
        "block_frames": blk,
        "duration_requested_s": args.duration,
        "duration_wall_s": elapsed,
        "n_control_samples": len(hist["e"]),
    }
    save_invasive_physical(hist, run_dir, meta)


if __name__ == "__main__":
    main()
