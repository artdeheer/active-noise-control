import os
import time

import numpy as np
import pyaudio

class IOManager:
    """
    Full-duplex I/O for a Behringer UMC204HD (or compatible UMC).

    Typical wiring for this project:
      - Input 1 (XLR/TRS): reference microphone
      - Input 2 (XLR/TRS): error microphone
      - Main L / Main R: loudspeakers (anti-noise is duplicated to both outputs)
    """

    def __init__(self, fs=48000, chunk=1):
        """
        fs: Requested sample rate. USB interfaces (e.g. UMC204HD) only support
            standard rates (44100 / 48000 / 96000, etc.); 2500 Hz from the paper
            will fail here — use simulation or add resampling for that case.
        chunk: Buffer size. Set to 1 for minimum latency in a short duct.
        """
        self.chunk = chunk
        self.p = pyaudio.PyAudio()

        self.device_index = self._find_umc204hd_device()
        if self.device_index is None:
            listing = self._duplex_device_listing()
            self.p.terminate()
            raise RuntimeError(
                "No Behringer UMC audio device found after several scans. "
                "Unplug/replug the interface, close other apps using it, or set "
                "ANC_PYAUDIO_DEVICE_INDEX to a duplex device index from a small "
                "listing script (see README or print devices in Python). "
                f"\n{listing}"
            )

        self.stream, self.fs = self._open_duplex_stream(preferred_rate=fs)

    def _duplex_device_listing(self, max_lines=24):
        """Human-readable duplex candidates for error messages."""
        lines = []
        for i in range(self.p.get_device_count()):
            dev = self.p.get_device_info_by_index(i)
            ins, outs = dev.get("maxInputChannels", 0), dev.get("maxOutputChannels", 0)
            if ins >= 2 and outs >= 2:
                lines.append(f"  [{i}] in={ins} out={outs} {dev['name']!r}")
            if len(lines) >= max_lines:
                lines.append("  …")
                break
        return "Duplex (≥2 in / ≥2 out) devices:\n" + (
            "\n".join(lines) if lines else "  (none reported)"
        )

    def _score_name(self, name_lower):
        score = 0
        if "204" in name_lower or "204hd" in name_lower:
            score += 10
        if "umc" in name_lower:
            score += 5
        if "behringer" in name_lower:
            score += 3
        return score

    def _find_umc204hd_device(self, attempts=8, delay_s=0.12):
        """Pick the UMC204HD (retry: PipeWire/ALSA sometimes omits USB nodes briefly)."""
        forced = os.environ.get("ANC_PYAUDIO_DEVICE_INDEX", "").strip()
        if forced.isdigit():
            idx = int(forced)
            if 0 <= idx < self.p.get_device_count():
                dev = self.p.get_device_info_by_index(idx)
                if dev.get("maxInputChannels", 0) >= 2 and dev.get("maxOutputChannels", 0) >= 2:
                    return idx

        for _ in range(attempts):
            chosen = self._scan_devices_once()
            if chosen is not None:
                return chosen
            time.sleep(delay_s)
        return None

    def _scan_devices_once(self):
        """Single enumeration pass; case-insensitive UMC / Behringer match."""
        candidates = []
        for i in range(self.p.get_device_count()):
            dev = self.p.get_device_info_by_index(i)
            name = dev["name"]
            nl = name.lower()
            if "umc" not in nl and "behringer" not in nl:
                continue
            if dev.get("maxInputChannels", 0) < 2 or dev.get("maxOutputChannels", 0) < 2:
                continue
            score = self._score_name(nl)
            candidates.append((score, i, name))

        if not candidates:
            return None
        candidates.sort(key=lambda t: (-t[0], t[1]))
        return candidates[0][1]

    def _open_duplex_stream(self, preferred_rate):
        """Try preferred rate, then common USB rates, until PortAudio accepts one."""
        seen = set()
        candidates = []
        for r in (preferred_rate, 48000, 44100, 96000, 32000, 16000):
            if r not in seen:
                seen.add(r)
                candidates.append(r)

        last_err = None
        for rate in candidates:
            try:
                stream = self.p.open(
                    format=pyaudio.paFloat32,
                    channels=2,  # In 1: Ref Mic, In 2: Error Mic
                    rate=rate,
                    input=True,
                    output=True,
                    input_device_index=self.device_index,
                    output_device_index=self.device_index,
                    frames_per_buffer=self.chunk,
                )
                return stream, rate
            except OSError as exc:
                last_err = exc

        self.p.terminate()
        raise RuntimeError(
            f"Could not open a duplex stream on device {self.device_index} "
            f"at any of {candidates} Hz. Last error: {last_err}"
        ) from last_err

    def capture_and_play(self, actuation_signal):
        """
        Reads one chunk of mic data and writes one chunk of anti-noise.
        Returns: r_n (reference), e_n (error)
        """
        # 1. Main L / Main R: same anti-noise on both outputs (stereo mains)
        out_data = np.array([actuation_signal, actuation_signal], dtype=np.float32)
        self.stream.write(out_data.tobytes())

        # 2. Capture inputs (Input 1: Ref, Input 2: Error)
        raw_in = self.stream.read(self.chunk, exception_on_overflow=False)
        in_samples = np.frombuffer(raw_in, dtype=np.float32)
        
        # Split the stereo interleaved data
        r_n = in_samples[0]
        e_n = in_samples[1]
        
        return r_n, e_n

    def exchange_block(self, actuation_mono: np.ndarray):
        """
        Full-duplex block I/O: play `len(actuation_mono)` stereo frames (mono duplicated
        to L/R), then read the same number of frames. `len(actuation_mono)` must equal
        self.chunk.
        Returns: reference mic r (n,), error mic e (n,)
        """
        actuation_mono = np.asarray(actuation_mono, dtype=np.float32).ravel()
        n = int(actuation_mono.shape[0])
        if n != self.chunk:
            raise ValueError(
                f"actuation_mono length {n} must match stream chunk size {self.chunk}"
            )
        stereo = np.empty(2 * n, dtype=np.float32)
        stereo[0::2] = actuation_mono
        stereo[1::2] = actuation_mono
        self.stream.write(stereo.tobytes())
        raw_in = self.stream.read(n, exception_on_overflow=False)
        inter = np.frombuffer(raw_in, dtype=np.float32).reshape(n, 2)
        return inter[:, 0].copy(), inter[:, 1].copy()

    def close(self):
        stream = getattr(self, "stream", None)
        if stream is not None:
            try:
                if stream.is_active():
                    stream.stop_stream()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
            self.stream = None
        pa = getattr(self, "p", None)
        if pa is not None:
            try:
                pa.terminate()
            except Exception:
                pass
            self.p = None