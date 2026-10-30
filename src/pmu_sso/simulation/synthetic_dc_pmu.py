"""Unchanged DFT helper extracted from the study waveform module."""
from __future__ import annotations
import math
import numpy as np
from ..config import SyntheticPMUChainConfig

def _sliding_dft_phasor(
    waveform: np.ndarray,
    *,
    sample_rate_hz: int,
    nominal_frequency_hz: float,
    report_rate_hz: int,
    chain: SyntheticPMUChainConfig,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(waveform, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("waveform must have shape (samples, 3)")
    samples_per_cycle = int(round(sample_rate_hz / nominal_frequency_hz))
    window_rows = chain.window_cycles * samples_per_cycle
    if len(values) < window_rows:
        raise ValueError("Waveform is shorter than the PMU estimation window")
    if chain.window_type == "rectangular":
        weights = np.ones(window_rows, dtype=float)
    else:
        weights = np.hanning(window_rows)
    time_s = np.arange(len(values), dtype=float) / sample_rate_hz
    demodulated = values * np.exp(-1j * 2 * np.pi * nominal_frequency_hz * time_s)[:, None]
    valid = np.column_stack(
        [np.convolve(demodulated[:, phase], weights, mode="valid") for phase in range(3)]
    )
    valid *= math.sqrt(2) / np.sum(weights)
    report_step = sample_rate_hz // report_rate_hz
    first_end = int(math.ceil((window_rows - 1) / report_step) * report_step)
    end_indices = np.arange(first_end, len(values), report_step, dtype=np.int64)
    start_indices = end_indices - window_rows + 1
    center_time = (start_indices + end_indices) / (2 * sample_rate_hz)
    return center_time, valid[start_indices]
