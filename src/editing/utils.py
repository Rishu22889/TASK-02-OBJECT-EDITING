"""Image I/O, timing, memory sampling and environment helpers."""

from __future__ import annotations

import json
import os
import platform
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import psutil
from PIL import Image


def read_image(path: str | Path) -> np.ndarray:
    """Read an image as BGR uint8. Falls back to Pillow for formats OpenCV lacks (e.g. AVIF)."""
    image = cv2.imread(str(path))
    if image is None:
        try:
            rgb = np.array(Image.open(path).convert("RGB"))
        except Exception as exc:
            raise FileNotFoundError(f"Cannot read image: {path}") from exc
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return image


def write_image(path: str | Path, image: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Cannot write image: {path}")
    return path


def configure_threads(threads: int = 0) -> int:
    """Use `threads` CPU threads in OpenCV and PyTorch (0 = all logical cores)."""
    threads = threads or os.cpu_count() or 1
    cv2.setNumThreads(threads)
    try:
        import torch

        torch.set_num_threads(threads)
    except ImportError:
        pass
    return threads


@contextmanager
def timed(timings: dict[str, float], key: str):
    """Add the wall-clock duration of the block to `timings[key]` (seconds)."""
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[key] = timings.get(key, 0.0) + time.perf_counter() - start


class PeakRSS:
    """Samples this process's RSS in a background thread; reports the peak inside a `with` block."""

    def __init__(self, interval: float = 0.01):
        self.interval = interval
        self.start = self.peak = 0

    def __enter__(self) -> "PeakRSS":
        self._proc = psutil.Process()
        self.start = self.peak = self._proc.memory_info().rss
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, self._proc.memory_info().rss)
            time.sleep(self.interval)

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self._proc.memory_info().rss)

    @property
    def peak_mb(self) -> float:
        return self.peak / 2**20

    @property
    def delta_mb(self) -> float:
        return (self.peak - self.start) / 2**20


def file_mb(path: str | Path) -> float:
    path = Path(path)
    if path.is_dir():
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 2**20
    return path.stat().st_size / 2**20


def fp32_mb(module) -> float:
    """In-memory fp32 weight size of a torch module (parameters x 4 bytes)."""
    return sum(p.numel() for p in module.parameters()) * 4 / 2**20


def cpu_model_name() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def system_info(threads: int) -> dict:
    info = {
        "cpu": cpu_model_name(),
        "logical_cores": os.cpu_count(),
        "threads_used": threads,
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 2),
        "python": platform.python_version(),
        "opencv": cv2.__version__,
    }
    for module in ("torch", "onnxruntime", "ultralytics", "diffusers", "transformers", "llama_cpp"):
        try:
            info[module] = __import__(module).__version__
        except Exception:
            pass
    return info


def save_json(path: str | Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
