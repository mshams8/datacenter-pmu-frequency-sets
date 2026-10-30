"""Physics-guided candidate scanning for aliased PMU oscillations.

The implementation is intentionally small and deterministic.  It assumes that
voltage angle and frequency are reported from the same PMU estimation chain.
That assumption is part of the measurement model and must not be generalized
to a PMU whose frequency channel is formed only from already-decimated angle
samples.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..response.estimator_response import EstimatorResponseModel


REQUIRED_CHANNELS = (
    "voltage_magnitude_pu",
    "voltage_angle_deg",
    "frequency_hz",
    "rocof_hz_per_s",
)

STATUS_INSUFFICIENT_DATA_QUALITY = "insufficient_data_quality"
STATUS_INSUFFICIENT_RESPONSE_SUPPORT = "insufficient_response_support"
STATUS_ESTIMATOR_RESPONSE_NOT_CERTIFIED = "estimator_response_not_certified"
STATUS_NO_CATALOGUE_EVENT = "no_catalogue_event"
STATUS_ALIAS_UNRESOLVED = "oscillation_detected_alias_unresolved"
STATUS_PHYSICAL_FREQUENCY_SET = "physical_frequency_set"

SPECTRAL_MAGNITUDE_FLOOR = 1e-15
DEFAULT_BACKGROUND_OFFSETS_HZ = (-1.5, -1.0, -0.6, 0.6, 1.0, 1.5)


def _decision_result(
    status: str,
    *,
    candidates: Sequence[float] = (),
    event_score: float | None = None,
    observed_alias_hz: float | None = None,
    response_calibration_id: str | None = None,
    reason: str | None = None,
) -> dict[str, object]:
    return {
        "status": status,
        "candidate_hz": [float(value) for value in candidates],
        "event_score": event_score,
        "observed_alias_hz": observed_alias_hz,
        "response_calibration_id": response_calibration_id,
        "reason": reason,
    }


def deterministic_quality_gate(
    time_s: np.ndarray,
    channels: Mapping[str, np.ndarray],
    report_rate_hz: float,
    *,
    expected_frame_count: int | None = None,
    frame_valid: np.ndarray | None = None,
    timestamp_tolerance_s: float = 1e-7,
) -> str | None:
    """Return a fail-closed status when reported data are incomplete or irregular.

    The gate does not interpolate, fill, or retime records.  Estimator and
    response identifiers are checked separately by the calibrated decision.
    """

    time = np.asarray(time_s, dtype=np.float64)
    expected_count = len(time) if expected_frame_count is None else int(expected_frame_count)
    if (
        time.ndim != 1
        or len(time) != expected_count
        or len(time) < 8
        or not np.all(np.isfinite(time))
        or not np.all(np.diff(time) > 0)
    ):
        return STATUS_INSUFFICIENT_DATA_QUALITY
    expected_step = 1.0 / float(report_rate_hz)
    if not np.allclose(
        np.diff(time), expected_step, rtol=0.0, atol=float(timestamp_tolerance_s)
    ):
        return STATUS_INSUFFICIENT_DATA_QUALITY
    if frame_valid is not None:
        valid = np.asarray(frame_valid)
        if valid.ndim != 1 or len(valid) != len(time) or not np.all(valid.astype(bool)):
            return STATUS_INSUFFICIENT_DATA_QUALITY
    for name in REQUIRED_CHANNELS:
        if name not in channels:
            return STATUS_INSUFFICIENT_DATA_QUALITY
        values = np.asarray(channels[name])
        if values.ndim != 1 or len(values) != len(time) or not np.all(np.isfinite(values)):
            return STATUS_INSUFFICIENT_DATA_QUALITY
    return None


def alias_frequency(frequency_hz: float, report_rate_hz: float) -> float:
    folded = float(frequency_hz) % float(report_rate_hz)
    return float(min(folded, float(report_rate_hz) - folded))


def select_max_support(
    support_by_frequency: Mapping[float, float],
    *,
    equality_tolerance: float = 1e-12,
) -> tuple[float, float]:
    """Select maximum support, using the lower frequency for numerical ties."""

    if not support_by_frequency:
        raise ValueError("At least one support value is required")
    maximum = max(float(value) for value in support_by_frequency.values())
    ties = sorted(
        float(frequency)
        for frequency, value in support_by_frequency.items()
        if math.isclose(
            float(value), maximum, rel_tol=0.0, abs_tol=float(equality_tolerance)
        )
    )
    return ties[0], maximum


def alias_class(
    observed_alias_hz: float,
    candidate_grid_hz: Sequence[float],
    report_rate_hz: float,
    tolerance_hz: float = 1e-9,
) -> np.ndarray:
    """Return physical-frequency candidates with the requested discrete-time fold."""
    grid = np.asarray(candidate_grid_hz, dtype=np.float64)
    aliases = np.asarray([alias_frequency(value, report_rate_hz) for value in grid])
    return grid[np.abs(aliases - float(observed_alias_hz)) <= float(tolerance_hz)]


def is_resolvable_fold(
    frequency_hz: float,
    report_rate_hz: float,
    tolerance_hz: float = 1e-9,
) -> bool:
    """Return false at the zero and exact-Nyquist folds.

    At either boundary the sampled sinusoid does not retain a general complex
    phase pair, so the estimator-specific amplitude ratio must not be used to
    force a physical-frequency choice.
    """
    folded = alias_frequency(frequency_hz, report_rate_hz)
    return bool(
        folded > float(tolerance_hz)
        and folded < float(report_rate_hz) / 2.0 - float(tolerance_hz)
    )


def detrend_linear(time_s: np.ndarray, values: np.ndarray) -> np.ndarray:
    time = np.asarray(time_s, dtype=np.float64)
    signal = np.asarray(values, dtype=np.float64)
    return signal - np.polyval(np.polyfit(time, signal, 1), time)


def unwrap_reported_angle_deg(values: np.ndarray) -> np.ndarray:
    """Unwrap a reported phase-angle sequence while retaining degree units."""

    angle_deg = np.asarray(values, dtype=np.float64)
    return np.rad2deg(np.unwrap(np.deg2rad(angle_deg)))


def complex_amplitude(time_s: np.ndarray, values: np.ndarray, frequency_hz: float) -> complex:
    """Return the normalized Hann-windowed coefficient after linear detrending.

    For samples ``y_n`` at times ``t_n``, the function evaluates
    ``2/sum(w_n) * sum(w_n * y_n * exp(-j 2 pi f t_n))``.  The factor of two
    gives the complex amplitude of a real sinusoid away from dc and Nyquist.
    """

    time = np.asarray(time_s, dtype=np.float64)
    centered = detrend_linear(time, values)
    window = np.hanning(len(centered))
    basis = np.exp(-2j * np.pi * float(frequency_hz) * time)
    return complex(2.0 * np.sum(centered * window * basis) / max(float(np.sum(window)), 1e-15))


def spectral_support_details(
    time_s: np.ndarray,
    values: np.ndarray,
    frequency_hz: float,
    report_rate_hz: float,
    *,
    background_offsets_hz: Sequence[float] = DEFAULT_BACKGROUND_OFFSETS_HZ,
) -> dict[str, float | int]:
    """Return target magnitude, local background, and their log ratio."""

    signal = abs(complex_amplitude(time_s, values, frequency_hz))
    comparisons = [
        abs(complex_amplitude(time_s, values, frequency_hz + offset))
        for offset in background_offsets_hz
        if 1.5 <= frequency_hz + offset <= report_rate_hz / 2.0
    ]
    noise = float(np.median(comparisons)) if comparisons else SPECTRAL_MAGNITUDE_FLOOR
    log_support = float(
        math.log(max(signal, SPECTRAL_MAGNITUDE_FLOOR) / max(noise, SPECTRAL_MAGNITUDE_FLOOR))
    )
    return {
        "coefficient_magnitude": float(signal),
        "background_magnitude": float(noise),
        "signal_background_ratio": float(
            max(signal, SPECTRAL_MAGNITUDE_FLOOR)
            / max(noise, SPECTRAL_MAGNITUDE_FLOOR)
        ),
        "log_support": log_support,
        "background_bin_count": len(comparisons),
    }


def local_log_support(
    time_s: np.ndarray,
    values: np.ndarray,
    frequency_hz: float,
    report_rate_hz: float,
    *,
    background_offsets_hz: Sequence[float] = DEFAULT_BACKGROUND_OFFSETS_HZ,
) -> float:
    return float(
        spectral_support_details(
            time_s,
            values,
            frequency_hz,
            report_rate_hz,
            background_offsets_hz=background_offsets_hz,
        )["log_support"]
    )


def scan_candidates(
    time_s: np.ndarray,
    channels: Mapping[str, np.ndarray],
    candidate_grid_hz: Sequence[float],
    report_rate_hz: float,
    response_model: "EstimatorResponseModel | None" = None,
    *,
    background_offsets_hz: Sequence[float] = DEFAULT_BACKGROUND_OFFSETS_HZ,
) -> list[dict[str, float]]:
    missing = [name for name in REQUIRED_CHANNELS if name not in channels]
    if missing:
        raise ValueError(f"Missing PMU channels: {missing}")
    prepared_channels = dict(channels)
    prepared_channels["voltage_angle_deg"] = unwrap_reported_angle_deg(
        channels["voltage_angle_deg"]
    )
    grid = np.asarray(candidate_grid_hz, dtype=np.float64)
    if grid.ndim != 1 or len(grid) == 0 or not np.all(np.isfinite(grid)):
        raise ValueError("candidate_grid_hz must be a finite nonempty vector")
    aliases = sorted({alias_frequency(value, report_rate_hz) for value in grid})
    alias_rows: dict[float, dict[str, float]] = {}
    for observed_hz in aliases:
        coefficients = {
            name: complex_amplitude(time_s, prepared_channels[name], observed_hz)
            for name in REQUIRED_CHANNELS
        }
        angle_amplitude = abs(coefficients["voltage_angle_deg"])
        frequency_amplitude = abs(coefficients["frequency_hz"])
        rocof_amplitude = abs(coefficients["rocof_hz_per_s"])
        supports = sorted(
            [
                local_log_support(
                    time_s,
                    prepared_channels[name],
                    observed_hz,
                    report_rate_hz,
                    background_offsets_hz=background_offsets_hz,
                )
                for name in REQUIRED_CHANNELS
            ],
            reverse=True,
        )
        alias_rows[observed_hz] = {
            "support": float(np.mean(supports[:2])),
            "angle_ratio_hz": float(180.0 / math.pi * frequency_amplitude / max(angle_amplitude, 1e-15)),
            "rocof_ratio_hz": float(
                rocof_amplitude / (2.0 * math.pi * max(frequency_amplitude, 1e-15))
            ),
            "frequency_log_amplitude": float(math.log(max(frequency_amplitude, 1e-15))),
        }
    output: list[dict[str, float]] = []
    for candidate_hz in grid:
        observed_hz = alias_frequency(float(candidate_hz), report_rate_hz)
        values = alias_rows[observed_hz]
        if response_model is None:
            expected_ratio_hz = float(candidate_hz)
            log_scale = 1.0
        else:
            response = response_model.response(float(candidate_hz))
            expected_ratio_hz = response.center_ratio_hz
            log_scale = response.log_scale
        angle_error = float(
            abs(
                math.log(
                    max(values["angle_ratio_hz"], 1e-9)
                    / max(expected_ratio_hz, 1e-9)
                )
            )
        )
        output.append(
            {
                "candidate_hz": float(candidate_hz),
                "observed_alias_hz": float(observed_hz),
                **values,
                "expected_angle_ratio_hz": float(expected_ratio_hz),
                "angle_log_scale": float(log_scale),
                "angle_error": angle_error,
                "response_nonconformity": float(angle_error / max(log_scale, 1e-12)),
                "rocof_error": float(
                    abs(math.log(max(values["rocof_ratio_hz"], 1e-9) / float(candidate_hz)))
                ),
            }
        )
    return output


def candidate_scores(candidate_rows: Sequence[Mapping[str, float]], penalty: float) -> np.ndarray:
    return np.asarray(
        [
            float(row["support"])
            - float(penalty)
            * float(row.get("response_nonconformity", row["angle_error"]))
            for row in candidate_rows
        ],
        dtype=np.float64,
    )


def selected_indices(scores: np.ndarray, delta: float) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    return np.flatnonzero(values >= np.max(values) - float(delta) - 1e-12)


def calibrated_candidate_decision(
    time_s: np.ndarray,
    channels: Mapping[str, np.ndarray],
    candidate_grid_hz: Sequence[float],
    report_rate_hz: float,
    *,
    response_model: "EstimatorResponseModel",
    estimator_configuration_id: str,
    response_calibration_id: str,
    event_threshold: float = -math.inf,
    expected_frame_count: int | None = None,
    frame_valid: np.ndarray | None = None,
) -> dict[str, object]:
    """Return a certified singleton, an unresolved candidate set, or abstention.

    A singleton is forbidden when the calibrated response interval overlaps an
    alias-compatible competitor, even if the current point estimate is closer
    to one candidate.  This makes estimator dependence an explicit part of the
    decision rather than an undocumented assumption.
    """

    refusal = deterministic_quality_gate(
        time_s,
        channels,
        report_rate_hz,
        expected_frame_count=expected_frame_count,
        frame_valid=frame_valid,
    )
    if refusal is not None:
        return _decision_result(refusal, reason="reported data failed deterministic quality checks")
    if (
        not response_model.certified
        or response_model.report_rate_hz != int(report_rate_hz)
        or response_model.estimator_configuration_id != estimator_configuration_id
        or response_model.calibration_id != response_calibration_id
    ):
        return _decision_result(
            STATUS_ESTIMATOR_RESPONSE_NOT_CERTIFIED,
            response_calibration_id=response_calibration_id,
            reason="estimator, report rate, or calibration identifier is not certified",
        )

    grid = np.asarray(candidate_grid_hz, dtype=np.float64)
    if set(np.round(grid, 9)) != set(np.round(response_model.candidate_grid_hz, 9)):
        return _decision_result(
            STATUS_ESTIMATOR_RESPONSE_NOT_CERTIFIED,
            response_calibration_id=response_calibration_id,
            reason="candidate catalogue does not match response calibration",
        )
    rows = scan_candidates(
        time_s,
        channels,
        grid,
        report_rate_hz,
        response_model=response_model,
    )
    alias_support: dict[float, float] = {}
    for row in rows:
        alias_support[float(row["observed_alias_hz"])] = max(
            alias_support.get(float(row["observed_alias_hz"]), -math.inf),
            float(row["support"]),
        )
    observed_alias_hz, event_score = select_max_support(alias_support)
    if event_score < float(event_threshold):
        return _decision_result(
            STATUS_NO_CATALOGUE_EVENT,
            event_score=float(event_score),
            observed_alias_hz=float(observed_alias_hz),
            response_calibration_id=response_model.calibration_id,
        )

    alias_rows = [
        row
        for row in rows
        if math.isclose(
            float(row["observed_alias_hz"]), float(observed_alias_hz), abs_tol=1e-9
        )
    ]
    alias_candidates = [float(row["candidate_hz"]) for row in alias_rows]
    if not alias_candidates:
        return _decision_result(
            STATUS_ESTIMATOR_RESPONSE_NOT_CERTIFIED,
            event_score=float(event_score),
            observed_alias_hz=float(observed_alias_hz),
            response_calibration_id=response_model.calibration_id,
            reason="no candidate is compatible with the detected alias",
        )
    if not is_resolvable_fold(alias_candidates[0], report_rate_hz):
        return _decision_result(
            STATUS_ALIAS_UNRESOLVED,
            candidates=alias_candidates,
            event_score=float(event_score),
            observed_alias_hz=float(observed_alias_hz),
            response_calibration_id=response_model.calibration_id,
            reason="zero or exact-Nyquist fold is not certified for a singleton",
        )

    observed_ratio_hz = float(alias_rows[0]["angle_ratio_hz"])
    compatible = [
        candidate
        for candidate in alias_candidates
        if response_model.compatible(candidate, observed_ratio_hz)
    ]
    if not compatible:
        return _decision_result(
            STATUS_ESTIMATOR_RESPONSE_NOT_CERTIFIED,
            candidates=alias_candidates,
            event_score=float(event_score),
            observed_alias_hz=float(observed_alias_hz),
            response_calibration_id=response_model.calibration_id,
            reason="observation lies outside every calibrated response interval",
        )
    if len(compatible) != 1:
        return _decision_result(
            STATUS_ALIAS_UNRESOLVED,
            candidates=compatible,
            event_score=float(event_score),
            observed_alias_hz=float(observed_alias_hz),
            response_calibration_id=response_model.calibration_id,
            reason="multiple alias-compatible responses contain the observation",
        )

    selected = compatible[0]
    overlapping = [
        candidate
        for candidate in alias_candidates
        if candidate != selected and response_model.intervals_overlap(selected, candidate)
    ]
    if overlapping:
        return _decision_result(
            STATUS_ALIAS_UNRESOLVED,
            candidates=sorted([selected, *overlapping]),
            event_score=float(event_score),
            observed_alias_hz=float(observed_alias_hz),
            response_calibration_id=response_model.calibration_id,
            reason="calibrated response regions overlap an alias-compatible competitor",
        )
    return _decision_result(
        STATUS_PHYSICAL_FREQUENCY_SET,
        candidates=[selected],
        event_score=float(event_score),
        observed_alias_hz=float(observed_alias_hz),
        response_calibration_id=response_model.calibration_id,
    )


def phase_folding_score(series: np.ndarray, report_rate_hz: float, phase_bins: int = 12) -> float:
    centered = np.asarray(series, dtype=np.float64) - float(np.mean(series))
    total = float(np.sum(centered**2))
    if len(centered) < phase_bins or total <= 1e-24:
        return 0.0
    power = np.abs(np.fft.rfft(centered * np.hanning(len(centered)))) ** 2
    frequency = np.fft.rfftfreq(len(centered), 1.0 / report_rate_hz)
    indices = np.flatnonzero((frequency >= 1.5) & (frequency <= report_rate_hz / 2.0))
    if not len(indices):
        return 0.0
    peak_hz = float(frequency[int(indices[np.argmax(power[indices])])])
    phase = (np.arange(len(centered), dtype=np.float64) * peak_hz / report_rate_hz) % 1.0
    bins = np.minimum((phase * phase_bins).astype(int), phase_bins - 1)
    reconstruction = np.zeros_like(centered)
    for phase_bin in range(phase_bins):
        selected = bins == phase_bin
        if np.any(selected):
            reconstruction[selected] = float(np.mean(centered[selected]))
    return float(np.clip(1.0 - np.sum((centered - reconstruction) ** 2) / total, 0.0, 1.0))


def global_spectral_score(series: np.ndarray, report_rate_hz: float) -> float:
    centered = np.asarray(series, dtype=np.float64) - float(np.mean(series))
    power = np.abs(np.fft.rfft(centered * np.hanning(len(centered)))) ** 2
    frequency = np.fft.rfftfreq(len(centered), 1.0 / report_rate_hz)
    band = (frequency >= 1.5) & (frequency <= report_rate_hz / 2.0)
    selected = power[band]
    if not len(selected):
        return 0.0
    return float(math.log1p(float(np.max(selected)) / max(float(np.median(selected)), 1e-24)))


def matched_baseline_scores(channels: Mapping[str, np.ndarray], report_rate_hz: float) -> dict[str, float]:
    folding = sorted(
        [phase_folding_score(channels[name], report_rate_hz) for name in REQUIRED_CHANNELS], reverse=True
    )
    spectral = [global_spectral_score(channels[name], report_rate_hz) for name in REQUIRED_CHANNELS]
    return {
        "physics_folding": float(np.mean(folding[:2])),
        "global_spectral": float(max(spectral)),
    }
