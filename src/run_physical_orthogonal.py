"""
Live orthogonal (non-invasive) ANC on the UMC interface.

Default: device audio at 48 kHz (or whatever opens), **controller + engine at
2.5 kHz** using polyphase resampling (48_000 × 5 / 96 = 2_500).

Run from ``src/``::

  cd src && uv run python run_physical_orthogonal.py --duration 30

Results: ``results/orthogonal/physical/<timestamp>/`` (plots + meta.json + scores.json).

``--no-decimate``: one control step per device sample (same rate as ``test.py``).
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

from io_manager import IOManager
from multirate import device_frames_per_process_block, poly_factors, resample_to_rate
from non_invasive.controller import NonInvasiveController
from non_invasive.engine import NonInvasiveEngine


def _load_config(path: Path):
    spec = importlib.util.spec_from_file_location("orth_physical_cfg", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def save_orthogonal_physical(history, run_dir, meta):
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    e = np.asarray(history["e"])
    r = np.asarray(history["r"])
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
    plt.title("Performance: ORTHOGONAL (physical)")
    plt.legend()
    plt.savefig(os.path.join(run_dir, "01_performance.png"))
    plt.close()
    print(f"Saved physical run to {run_dir}")


def main():
    parser = argparse.ArgumentParser(description="Orthogonal ANC on hardware.")
    parser.add_argument("--duration", type=float, default=30.0, help="Seconds.")
    parser.add_argument(
        "--no-decimate",
        action="store_true",
        help="Run the controller at the device sample rate (no 2.5 kHz conversion).",
    )
    parser.add_argument(
        "--name",
        "-n",
        type=str,
        default=None,
        help="Subfolder under results/orthogonal/physical/",
    )
    args = parser.parse_args()

    orth_cfg = _load_config(SRC / "non_invasive" / "config.py")
    orth_process_fs = int(orth_cfg.FS)

    io = IOManager(fs=48000, chunk=1)
    device_fs = int(io.fs)
    decimate = not args.no_decimate
    if decimate:
        blk = device_frames_per_process_block(device_fs, orth_process_fs)
        up, dn = poly_factors(device_fs, orth_process_fs)
        approx_fs = device_fs * up / dn
        if abs(approx_fs - orth_process_fs) > 1e-3:
            io.close()
            raise SystemExit(
                f"No usable polyphase ratio between device_fs={device_fs} Hz and "
                f"non_invasive/config.FS={orth_process_fs} Hz (would be {approx_fs:.4f} Hz). "
                "Use --no-decimate or an interface at 48000 Hz for exact 2.5 kHz decimation."
            )
        if blk != 1:
            io.close()
            io = IOManager(fs=device_fs, chunk=blk)
        control_fs = orth_process_fs
    else:
        blk = 1
        control_fs = device_fs

    m_taps = orth_cfg.S_TAPS
    engine = NonInvasiveEngine(m=m_taps, rng=np.random.default_rng(orth_cfg.RANDOM_SEED))
    controller = NonInvasiveController(
        m=m_taps,
        startup_samples=orth_cfg.STARTUP_PROTECTION_SAMPLES,
        startup_actuation_limit=orth_cfg.STARTUP_ACTUATION_LIMIT,
        update_alpha=orth_cfg.CONTROLLER_UPDATE_ALPHA,
        max_actuation_step=orth_cfg.MAX_ACTUATION_STEP,
    )

    n_blocks = max(1, int(args.duration * device_fs / blk))
    hist = {"e": [], "r": [], "a": []}
    last_a = 0.0
    out_high = np.zeros(blk, dtype=np.float32)

    print(
        f"Orthogonal physical | device_fs={device_fs} Hz | block={blk} | "
        f"decimate={'on' if decimate else 'off'} | control_fs≈{control_fs} Hz | "
        f"blocks≈{n_blocks} (Ctrl+C stops early)"
    )
    t0 = time.time()
    try:
        for _ in range(n_blocks):
            r_hi, e_hi = io.exchange_block(out_high)
            if decimate:
                r_lo = resample_to_rate(r_hi, device_fs, orth_cfg.FS)
                e_lo = resample_to_rate(e_hi, device_fs, orth_cfg.FS)
            else:
                r_lo, e_lo = r_hi.astype(float), e_hi.astype(float)
            a_block = []
            for i in range(len(r_lo)):
                r_n, e_n = float(r_lo[i]), float(e_lo[i])
                h_hat, s_hat = engine.update(r_n, last_a, e_n)
                if int(time.time() * 10) % 2 == 0:
                    controller.solve_or_update(h_hat, s_hat)
                a_n = controller.generate_actuation(r_n)
                last_a = a_n
                hist["r"].append(r_n)
                hist["e"].append(e_n)
                hist["a"].append(a_n)
                a_block.append(a_n)
            if decimate:
                out_high = resample_to_rate(
                    np.asarray(a_block, dtype=np.float64), orth_cfg.FS, device_fs
                ).astype(np.float32)
            else:
                out_high = np.asarray(a_block, dtype=np.float32)
    except KeyboardInterrupt:
        print("\nStopped early.")
    finally:
        io.close()

    if not hist["e"]:
        print("No samples captured; not writing results.")
        return
    elapsed = time.time() - t0
    folder = args.name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("results", "orthogonal", "physical", folder)
    meta = {
        "algorithm": "orthogonal",
        "mode": "physical",
        "device_fs_hz": device_fs,
        "process_fs_hz": orth_cfg.FS if decimate else device_fs,
        "decimated": decimate,
        "block_frames": blk,
        "duration_requested_s": args.duration,
        "duration_wall_s": elapsed,
        "n_control_samples": len(hist["e"]),
        "s_taps": m_taps,
    }
    save_orthogonal_physical(hist, run_dir, meta)


if __name__ == "__main__":
    main()
