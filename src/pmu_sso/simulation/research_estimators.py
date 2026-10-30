"""Transparent research synchrophasor estimators for response studies.

These implementations expose their windows and frequency-estimation stages so
that estimator dependence can be measured.  They are research estimators, not
claims of IEEE/IEC conformance or commercial-PMU behavior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.signal import resample_poly

from ..config import SyntheticPMUChainConfig
from .synthetic_dc_pmu import _sliding_dft_phasor


@dataclass(frozen=True)
class ResearchPMUChannels:
    """Reported positive-sequence channels and their estimator identity."""

    timestamp_s: np.ndarray
    channels: Mapping[str, np.ndarray]
    estimator_configuration_id: str
    estimator_kind: str
    frequency_stage: str
    report_rate_hz: int

    def __post_init__(self) -> None:
        time = np.asarray(self.timestamp_s, dtype=np.float64)
        if time.ndim != 1 or len(time) < 8 or not np.all(np.isfinite(time)):
            raise ValueError("timestamp_s must be a finite vector with at least eight reports")
        if not np.all(np.diff(time) > 0):
            raise ValueError("timestamp_s must be strictly increasing")
        for name in (
            "voltage_magnitude_pu",
            "voltage_angle_deg",
            "frequency_hz",
            "rocof_hz_per_s",
        ):
            if name not in self.channels:
                raise ValueError(f"Missing research-estimator channel: {name}")
            values = np.asarray(self.channels[name], dtype=np.float64)
            if values.shape != time.shape or not np.all(np.isfinite(values)):
                raise ValueError(f"Channel {name} must be finite and aligned to timestamp_s")
        object.__setattr__(self, "timestamp_s", time)


@dataclass(frozen=True)
class ScalarResearchPMUChannels:
    """Generic scalar-channel synchrophasor output for waveform replay."""

    timestamp_s: np.ndarray
    complex_phasor: np.ndarray
    channels: Mapping[str, np.ndarray]
    estimator_configuration_id: str
    estimator_kind: str
    frequency_stage: str
    report_rate_hz: int
    nominal_frequency_hz: float
    signal_kind: str

    def __post_init__(self) -> None:
        time = np.asarray(self.timestamp_s, dtype=np.float64)
        phasor = np.asarray(self.complex_phasor, dtype=np.complex128)
        if time.ndim != 1 or len(time) < 8 or not np.all(np.isfinite(time)):
            raise ValueError("timestamp_s must be a finite vector with at least eight reports")
        if not np.all(np.diff(time) > 0):
            raise ValueError("timestamp_s must be strictly increasing")
        if phasor.shape != time.shape or not np.all(np.isfinite(phasor)):
            raise ValueError("complex_phasor must be finite and aligned to timestamp_s")
        for name in (
            "phasor_magnitude_pu",
            "phasor_angle_deg",
            "frequency_hz",
            "rocof_hz_per_s",
        ):
            if name not in self.channels:
                raise ValueError(f"Missing scalar research-estimator channel: {name}")
            values = np.asarray(self.channels[name], dtype=np.float64)
            if values.shape != time.shape or not np.all(np.isfinite(values)):
                raise ValueError(f"Channel {name} must be finite and aligned to timestamp_s")
        object.__setattr__(self, "timestamp_s", time)
        object.__setattr__(self, "complex_phasor", phasor)


@dataclass(frozen=True)
class PolyphaseResampleResult:
    """A deterministic scalar resampling result with explicit provenance."""

    timestamp_s: np.ndarray
    waveform: np.ndarray
    input_sample_rate_hz: int
    output_sample_rate_hz: int
    up: int
    down: int
    window: tuple[str, float]
    padtype: str


def estimator_configuration_id(chain: SyntheticPMUChainConfig, report_rate_hz: int) -> str:
    """Return the complete response-critical estimator/configuration identifier."""

    taylor_suffix = (
        f"__order{chain.taylor_order}__ridge{chain.taylor_ridge_relative:.3g}"
        if chain.estimator_kind == "taylor_fourier"
        else ""
    )
    return (
        f"{chain.chain_id}__{chain.estimator_kind}__{chain.frequency_stage}__"
        f"{chain.window_type}{chain.window_cycles}cycle__{chain.front_end_filter_id}__"
        f"{int(report_rate_hz)}fps{taylor_suffix}__{chain.implementation_version}"
    )


def _positive_sequence(phase_phasors: np.ndarray) -> np.ndarray:
    values = np.asarray(phase_phasors, dtype=np.complex128)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("phase_phasors must have shape (reports, 3)")
    rotation = np.exp(2j * np.pi / 3)
    return (values[:, 0] + rotation * values[:, 1] + rotation**2 * values[:, 2]) / 3.0


def _reported_channels(
    time_s: np.ndarray,
    positive_phasor: np.ndarray,
    frequency_hz: np.ndarray,
    nominal_frequency_hz: float,
) -> dict[str, np.ndarray]:
    time = np.asarray(time_s, dtype=np.float64)
    phasor = np.asarray(positive_phasor, dtype=np.complex128)
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    if phasor.shape != time.shape or frequency.shape != time.shape:
        raise ValueError("Phasor and frequency vectors must align to timestamps")
    angle_rad = np.unwrap(np.angle(phasor))
    magnitude = np.abs(phasor)
    magnitude_pu = magnitude / max(float(np.median(magnitude)), np.finfo(float).tiny)
    rocof = np.gradient(frequency, time, edge_order=2)
    return {
        "voltage_magnitude_pu": magnitude_pu,
        "voltage_angle_deg": np.rad2deg(angle_rad),
        "frequency_hz": nominal_frequency_hz + (frequency - nominal_frequency_hz),
        "rocof_hz_per_s": rocof,
    }


def _scalar_reported_channels(
    time_s: np.ndarray,
    phasor: np.ndarray,
    frequency_hz: np.ndarray,
) -> dict[str, np.ndarray]:
    time = np.asarray(time_s, dtype=np.float64)
    phasor_values = np.asarray(phasor, dtype=np.complex128)
    frequency = np.asarray(frequency_hz, dtype=np.float64)
    if phasor_values.shape != time.shape or frequency.shape != time.shape:
        raise ValueError("Scalar phasor and frequency vectors must align to timestamps")
    magnitude = np.abs(phasor_values)
    magnitude_scale = max(float(np.median(magnitude)), np.finfo(float).tiny)
    return {
        "phasor_magnitude_pu": magnitude / magnitude_scale,
        "phasor_angle_deg": np.rad2deg(np.unwrap(np.angle(phasor_values))),
        "frequency_hz": frequency,
        "rocof_hz_per_s": np.gradient(frequency, time, edge_order=2),
    }


def resample_scalar_waveform_polyphase(
    waveform: np.ndarray,
    *,
    input_sample_rate_hz: int,
    output_sample_rate_hz: int = 6_000,
    window: tuple[str, float] = ("kaiser", 5.0),
    padtype: str = "line",
) -> PolyphaseResampleResult:
    """Resample one real waveform on a registered uniform grid.

    SciPy compensates the symmetric FIR group delay so the first output sample
    shares the input time origin.  Boundary samples remain filter-dependent and
    must be excluded by the experiment configuration.
    """

    values = np.asarray(waveform, dtype=np.float64)
    if values.ndim != 1 or len(values) < 8 or not np.all(np.isfinite(values)):
        raise ValueError("waveform must be a finite vector with at least eight samples")
    if input_sample_rate_hz <= 0 or output_sample_rate_hz <= 0:
        raise ValueError("sample rates must be positive")
    common = math.gcd(int(input_sample_rate_hz), int(output_sample_rate_hz))
    up = int(output_sample_rate_hz) // common
    down = int(input_sample_rate_hz) // common
    resampled = resample_poly(values, up, down, window=window, padtype=padtype)
    time_s = np.arange(len(resampled), dtype=np.float64) / output_sample_rate_hz
    return PolyphaseResampleResult(
        timestamp_s=time_s,
        waveform=np.asarray(resampled, dtype=np.float64),
        input_sample_rate_hz=int(input_sample_rate_hz),
        output_sample_rate_hz=int(output_sample_rate_hz),
        up=up,
        down=down,
        window=window,
        padtype=padtype,
    )


def _sliding_dft_scalar_phasor(
    waveform: np.ndarray,
    *,
    sample_rate_hz: int,
    nominal_frequency_hz: float,
    report_rate_hz: int,
    chain: SyntheticPMUChainConfig,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(waveform, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("scalar waveform must be one-dimensional")
    if sample_rate_hz % report_rate_hz != 0:
        raise ValueError("sample_rate_hz must be divisible by report_rate_hz")
    samples_per_cycle = int(round(sample_rate_hz / nominal_frequency_hz))
    window_rows = chain.window_cycles * samples_per_cycle
    if len(values) < window_rows:
        raise ValueError("Waveform is shorter than the scalar PMU estimation window")
    weights = _window_weights(window_rows, chain.window_type)
    source_time_s = np.arange(len(values), dtype=np.float64) / sample_rate_hz
    demodulated = values * np.exp(
        -1j * 2 * np.pi * nominal_frequency_hz * source_time_s
    )
    valid = np.convolve(demodulated, weights, mode="valid")
    valid *= math.sqrt(2) / np.sum(weights)
    report_step = sample_rate_hz // report_rate_hz
    first_end = int(math.ceil((window_rows - 1) / report_step) * report_step)
    end_indices = np.arange(first_end, len(values), report_step, dtype=np.int64)
    start_indices = end_indices - window_rows + 1
    center_time = (start_indices + end_indices) / (2 * sample_rate_hz)
    return center_time, np.asarray(valid[start_indices], dtype=np.complex128)


def _sliding_dft_scalar_channels(
    waveform: np.ndarray,
    *,
    sample_rate_hz: int,
    nominal_frequency_hz: float,
    report_rate_hz: int,
    chain: SyntheticPMUChainConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    report_time, reported_phasor = _sliding_dft_scalar_phasor(
        waveform,
        sample_rate_hz=sample_rate_hz,
        nominal_frequency_hz=nominal_frequency_hz,
        report_rate_hz=report_rate_hz,
        chain=chain,
    )
    if chain.frequency_stage == "reported_phase_derivative":
        angle = np.unwrap(np.angle(reported_phasor))
        frequency = nominal_frequency_hz + np.gradient(
            angle, report_time, edge_order=2
        ) / (2 * np.pi)
    else:
        high_time, high_phasor = _sliding_dft_scalar_phasor(
            waveform,
            sample_rate_hz=sample_rate_hz,
            nominal_frequency_hz=nominal_frequency_hz,
            report_rate_hz=sample_rate_hz,
            chain=chain,
        )
        high_angle = np.unwrap(np.angle(high_phasor))
        high_frequency = nominal_frequency_hz + np.gradient(
            high_angle, high_time, edge_order=2
        ) / (2 * np.pi)
        frequency = np.interp(report_time, high_time, high_frequency)
    return (
        report_time,
        reported_phasor,
        _scalar_reported_channels(report_time, reported_phasor, frequency),
    )


def _sliding_dft_channels(
    waveform_v: np.ndarray,
    *,
    sample_rate_hz: int,
    nominal_frequency_hz: float,
    report_rate_hz: int,
    chain: SyntheticPMUChainConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    report_time, reported_phases = _sliding_dft_phasor(
        waveform_v,
        sample_rate_hz=sample_rate_hz,
        nominal_frequency_hz=nominal_frequency_hz,
        report_rate_hz=report_rate_hz,
        chain=chain,
    )
    reported_positive = _positive_sequence(reported_phases)
    if chain.frequency_stage == "reported_phase_derivative":
        angle = np.unwrap(np.angle(reported_positive))
        frequency = nominal_frequency_hz + np.gradient(angle, report_time, edge_order=2) / (
            2 * np.pi
        )
        return report_time, _reported_channels(
            report_time, reported_positive, frequency, nominal_frequency_hz
        )

    high_time, high_phases = _sliding_dft_phasor(
        waveform_v,
        sample_rate_hz=sample_rate_hz,
        nominal_frequency_hz=nominal_frequency_hz,
        report_rate_hz=sample_rate_hz,
        chain=chain,
    )
    high_positive = _positive_sequence(high_phases)
    high_angle = np.unwrap(np.angle(high_positive))
    high_frequency = nominal_frequency_hz + np.gradient(
        high_angle, high_time, edge_order=2
    ) / (2 * np.pi)
    report_frequency = np.interp(report_time, high_time, high_frequency)
    return report_time, _reported_channels(
        report_time, reported_positive, report_frequency, nominal_frequency_hz
    )


def _window_weights(window_rows: int, window_type: str) -> np.ndarray:
    return (
        np.ones(window_rows, dtype=np.float64)
        if window_type == "rectangular"
        else np.hanning(window_rows)
    )


def _taylor_fourier_phasor(
    waveform_v: np.ndarray,
    *,
    sample_rate_hz: int,
    nominal_frequency_hz: float,
    report_rate_hz: int,
    chain: SyntheticPMUChainConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a declared-order Taylor--Fourier phasor at each report instant.

    The real waveform is represented over a declared window by a complex RMS
    phasor and its first time derivative.  Frequency is obtained from
    ``imag(V1 / V0)/(2*pi)`` rather than from already-decimated phase samples.
    Higher-order terms reduce truncation bias; an optional relative ridge term
    stabilizes the weighted normal equations without penalizing the base
    phasor.  The fitted coefficients are rotated to the global nominal-frequency
    reference, which is required when reports do not occur an integer number of
    nominal cycles apart.
    """

    values = np.asarray(waveform_v, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("waveform_v must have shape (samples, 3)")
    if sample_rate_hz % report_rate_hz != 0:
        raise ValueError("sample_rate_hz must be divisible by report_rate_hz")
    samples_per_cycle = int(round(sample_rate_hz / nominal_frequency_hz))
    window_rows = chain.window_cycles * samples_per_cycle
    if len(values) < window_rows:
        raise ValueError("Waveform is shorter than the Taylor--Fourier window")

    relative_time_s = (
        np.arange(window_rows, dtype=np.float64) - (window_rows - 1) / 2.0
    ) / sample_rate_hz
    time_scale_s = max(float(np.max(np.abs(relative_time_s))), 1.0 / sample_rate_hz)
    normalized_time = relative_time_s / time_scale_s
    absolute_time = np.arange(window_rows, dtype=np.float64) / sample_rate_hz
    omega_time = 2 * np.pi * nominal_frequency_hz * absolute_time
    carrier_columns: list[np.ndarray] = []
    for order in range(chain.taylor_order + 1):
        polynomial = normalized_time**order / math.factorial(order)
        carrier_columns.extend(
            (polynomial * np.cos(omega_time), polynomial * np.sin(omega_time))
        )
    design = np.column_stack(carrier_columns)
    weights = _window_weights(window_rows, chain.window_type)
    weighted_design = design * np.sqrt(weights)[:, None]
    if chain.taylor_ridge_relative > 0:
        gram = weighted_design.T @ weighted_design
        penalty = chain.taylor_ridge_relative * np.diag(np.diag(gram))
        penalty[:2, :2] = 0.0
        projection = np.linalg.solve(
            gram + penalty,
            weighted_design.T,
        ) * np.sqrt(weights)[None, :]
    else:
        projection = np.linalg.pinv(weighted_design) * np.sqrt(weights)[None, :]
    coefficient_count = 2 * (chain.taylor_order + 1)
    coefficient_series = np.empty(
        (len(values) - window_rows + 1, 3, coefficient_count), dtype=np.float64
    )
    for phase in range(3):
        for coefficient in range(coefficient_count):
            coefficient_series[:, phase, coefficient] = np.convolve(
                values[:, phase], projection[coefficient, ::-1], mode="valid"
            )

    report_step = sample_rate_hz // report_rate_hz
    first_end = int(math.ceil((window_rows - 1) / report_step) * report_step)
    end_indices = np.arange(first_end, len(values), report_step, dtype=np.int64)
    start_indices = end_indices - window_rows + 1
    selected = coefficient_series[start_indices]
    phasor = (selected[:, :, 0] - 1j * selected[:, :, 1]) / math.sqrt(2)
    derivative = (
        (selected[:, :, 2] - 1j * selected[:, :, 3])
        / (math.sqrt(2) * time_scale_s)
    )
    reference_rotation = np.exp(
        -1j * 2 * np.pi * nominal_frequency_hz * start_indices / sample_rate_hz
    )
    phasor *= reference_rotation[:, None]
    derivative *= reference_rotation[:, None]
    center_time = (start_indices + end_indices) / (2 * sample_rate_hz)
    return center_time, phasor, derivative


def _taylor_fourier_channels(
    waveform_v: np.ndarray,
    *,
    sample_rate_hz: int,
    nominal_frequency_hz: float,
    report_rate_hz: int,
    chain: SyntheticPMUChainConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    time_s, phase_phasors, phase_derivatives = _taylor_fourier_phasor(
        waveform_v,
        sample_rate_hz=sample_rate_hz,
        nominal_frequency_hz=nominal_frequency_hz,
        report_rate_hz=report_rate_hz,
        chain=chain,
    )
    positive = _positive_sequence(phase_phasors)
    positive_derivative = _positive_sequence(phase_derivatives)
    ratio = positive_derivative / np.where(
        np.abs(positive) > np.finfo(float).tiny,
        positive,
        np.finfo(float).tiny + 0j,
    )
    frequency = nominal_frequency_hz + np.imag(ratio) / (2 * np.pi)
    return time_s, _reported_channels(time_s, positive, frequency, nominal_frequency_hz)


def _taylor_fourier_scalar_phasor(
    waveform: np.ndarray,
    *,
    sample_rate_hz: int,
    nominal_frequency_hz: float,
    report_rate_hz: int,
    chain: SyntheticPMUChainConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(waveform, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("scalar waveform must be one-dimensional")
    if sample_rate_hz % report_rate_hz != 0:
        raise ValueError("sample_rate_hz must be divisible by report_rate_hz")
    samples_per_cycle = int(round(sample_rate_hz / nominal_frequency_hz))
    window_rows = chain.window_cycles * samples_per_cycle
    if len(values) < window_rows:
        raise ValueError("Waveform is shorter than the Taylor--Fourier window")

    relative_time_s = (
        np.arange(window_rows, dtype=np.float64) - (window_rows - 1) / 2.0
    ) / sample_rate_hz
    time_scale_s = max(float(np.max(np.abs(relative_time_s))), 1.0 / sample_rate_hz)
    normalized_time = relative_time_s / time_scale_s
    absolute_time = np.arange(window_rows, dtype=np.float64) / sample_rate_hz
    omega_time = 2 * np.pi * nominal_frequency_hz * absolute_time
    carrier_columns: list[np.ndarray] = []
    for order in range(chain.taylor_order + 1):
        polynomial = normalized_time**order / math.factorial(order)
        carrier_columns.extend(
            (polynomial * np.cos(omega_time), polynomial * np.sin(omega_time))
        )
    design = np.column_stack(carrier_columns)
    weights = _window_weights(window_rows, chain.window_type)
    weighted_design = design * np.sqrt(weights)[:, None]
    if chain.taylor_ridge_relative > 0:
        gram = weighted_design.T @ weighted_design
        penalty = chain.taylor_ridge_relative * np.diag(np.diag(gram))
        penalty[:2, :2] = 0.0
        projection = np.linalg.solve(gram + penalty, weighted_design.T) * np.sqrt(
            weights
        )[None, :]
    else:
        projection = np.linalg.pinv(weighted_design) * np.sqrt(weights)[None, :]
    coefficient_count = 2 * (chain.taylor_order + 1)
    coefficient_series = np.empty(
        (len(values) - window_rows + 1, coefficient_count), dtype=np.float64
    )
    for coefficient in range(coefficient_count):
        coefficient_series[:, coefficient] = np.convolve(
            values, projection[coefficient, ::-1], mode="valid"
        )

    report_step = sample_rate_hz // report_rate_hz
    first_end = int(math.ceil((window_rows - 1) / report_step) * report_step)
    end_indices = np.arange(first_end, len(values), report_step, dtype=np.int64)
    start_indices = end_indices - window_rows + 1
    selected = coefficient_series[start_indices]
    phasor = (selected[:, 0] - 1j * selected[:, 1]) / math.sqrt(2)
    derivative = (
        (selected[:, 2] - 1j * selected[:, 3]) / (math.sqrt(2) * time_scale_s)
    )
    reference_rotation = np.exp(
        -1j * 2 * np.pi * nominal_frequency_hz * start_indices / sample_rate_hz
    )
    phasor *= reference_rotation
    derivative *= reference_rotation
    center_time = (start_indices + end_indices) / (2 * sample_rate_hz)
    return center_time, phasor, derivative


def _taylor_fourier_scalar_channels(
    waveform: np.ndarray,
    *,
    sample_rate_hz: int,
    nominal_frequency_hz: float,
    report_rate_hz: int,
    chain: SyntheticPMUChainConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    time_s, phasor, derivative = _taylor_fourier_scalar_phasor(
        waveform,
        sample_rate_hz=sample_rate_hz,
        nominal_frequency_hz=nominal_frequency_hz,
        report_rate_hz=report_rate_hz,
        chain=chain,
    )
    ratio = derivative / np.where(
        np.abs(phasor) > np.finfo(float).tiny,
        phasor,
        np.finfo(float).tiny + 0j,
    )
    frequency = nominal_frequency_hz + np.imag(ratio) / (2 * np.pi)
    return time_s, phasor, _scalar_reported_channels(time_s, phasor, frequency)


def estimate_research_pmu_channels(
    waveform_v: np.ndarray,
    *,
    sample_rate_hz: int,
    nominal_frequency_hz: float,
    report_rate_hz: int,
    chain: SyntheticPMUChainConfig,
) -> ResearchPMUChannels:
    """Estimate transparent reported channels from one three-phase waveform."""

    if sample_rate_hz % int(report_rate_hz) != 0:
        raise ValueError("sample_rate_hz must be divisible by report_rate_hz")
    if chain.estimator_kind == "sliding_dft":
        time_s, channels = _sliding_dft_channels(
            waveform_v,
            sample_rate_hz=sample_rate_hz,
            nominal_frequency_hz=nominal_frequency_hz,
            report_rate_hz=report_rate_hz,
            chain=chain,
        )
    elif chain.estimator_kind == "taylor_fourier":
        time_s, channels = _taylor_fourier_channels(
            waveform_v,
            sample_rate_hz=sample_rate_hz,
            nominal_frequency_hz=nominal_frequency_hz,
            report_rate_hz=report_rate_hz,
            chain=chain,
        )
    else:  # pragma: no cover - Pydantic validation prevents this branch.
        raise ValueError(f"Unsupported estimator kind: {chain.estimator_kind}")
    return ResearchPMUChannels(
        timestamp_s=time_s,
        channels=channels,
        estimator_configuration_id=estimator_configuration_id(chain, report_rate_hz),
        estimator_kind=chain.estimator_kind,
        frequency_stage=chain.frequency_stage,
        report_rate_hz=int(report_rate_hz),
    )


def estimate_scalar_research_pmu_channels(
    waveform: np.ndarray,
    *,
    sample_rate_hz: int,
    nominal_frequency_hz: float,
    report_rate_hz: int,
    chain: SyntheticPMUChainConfig,
    signal_kind: str = "current",
) -> ScalarResearchPMUChannels:
    """Estimate transparent scalar synchrophasor channels from one waveform.

    The estimator identity is still configuration-specific.  Passing a scalar
    field current does not imply positive-sequence processing or commercial-PMU
    equivalence.
    """

    if sample_rate_hz % int(report_rate_hz) != 0:
        raise ValueError("sample_rate_hz must be divisible by report_rate_hz")
    if signal_kind not in {"current", "voltage", "generic"}:
        raise ValueError("signal_kind must be current, voltage, or generic")
    if chain.estimator_kind == "sliding_dft":
        time_s, phasor, channels = _sliding_dft_scalar_channels(
            waveform,
            sample_rate_hz=sample_rate_hz,
            nominal_frequency_hz=nominal_frequency_hz,
            report_rate_hz=report_rate_hz,
            chain=chain,
        )
    elif chain.estimator_kind == "taylor_fourier":
        time_s, phasor, channels = _taylor_fourier_scalar_channels(
            waveform,
            sample_rate_hz=sample_rate_hz,
            nominal_frequency_hz=nominal_frequency_hz,
            report_rate_hz=report_rate_hz,
            chain=chain,
        )
    else:  # pragma: no cover - Pydantic validation prevents this branch.
        raise ValueError(f"Unsupported estimator kind: {chain.estimator_kind}")
    scalar_id = estimator_configuration_id(chain, report_rate_hz).replace(
        f"__{int(report_rate_hz)}fps", f"__scalar_{signal_kind}__{int(report_rate_hz)}fps"
    )
    return ScalarResearchPMUChannels(
        timestamp_s=time_s,
        complex_phasor=phasor,
        channels=channels,
        estimator_configuration_id=scalar_id,
        estimator_kind=chain.estimator_kind,
        frequency_stage=chain.frequency_stage,
        report_rate_hz=int(report_rate_hz),
        nominal_frequency_hz=float(nominal_frequency_hz),
        signal_kind=signal_kind,
    )


__all__ = [
    "PolyphaseResampleResult",
    "ResearchPMUChannels",
    "ScalarResearchPMUChannels",
    "estimate_research_pmu_channels",
    "estimate_scalar_research_pmu_channels",
    "estimator_configuration_id",
    "resample_scalar_waveform_polyphase",
]
