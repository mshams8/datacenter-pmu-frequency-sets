"""Complex multichannel response calibration for exact alias classes.

The response is intentionally estimator-specific.  It uses amplitude and phase
relations among angle, frequency, and ROCOF at the observed folded frequency;
it does not claim that a response learned for one PMU chain transfers to a
different estimator or front-end filter.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ..inference.calibration import conformal_order_statistic
from ..inference.candidate_scan import (
    REQUIRED_CHANNELS,
    STATUS_ALIAS_UNRESOLVED,
    STATUS_ESTIMATOR_RESPONSE_NOT_CERTIFIED,
    STATUS_INSUFFICIENT_RESPONSE_SUPPORT,
    STATUS_NO_CATALOGUE_EVENT,
    STATUS_PHYSICAL_FREQUENCY_SET,
    _decision_result,
    alias_frequency,
    complex_amplitude,
    deterministic_quality_gate,
    is_resolvable_fold,
    scan_candidates,
    select_max_support,
    spectral_support_details,
    unwrap_reported_angle_deg,
)
from ..provenance import stable_hash


COMPLEX_RESPONSE_FEATURES = (
    "response_log_frequency_angle",
    "response_frequency_angle_cos",
    "response_frequency_angle_sin",
    "response_log_rocof_frequency",
    "response_rocof_frequency_cos",
    "response_rocof_frequency_sin",
)

RESPONSE_LOG_MAGNITUDE_FLOOR = 1e-15
RESPONSE_CHANNEL_SUPPORT_THRESHOLD = math.log(2.0)
# Backward-compatible name retained for archived configurations and callers.
RESPONSE_DENOMINATOR_SUPPORT_THRESHOLD = RESPONSE_CHANNEL_SUPPORT_THRESHOLD


def response_support_diagnostics(
    time_s: np.ndarray,
    channels: Mapping[str, np.ndarray],
    observed_alias_hz: float,
    report_rate_hz: float,
    *,
    background_offsets_hz: Sequence[float] = (-1.5, -1.0, -0.6, 0.6, 1.0, 1.5),
    minimum_log_support: float = RESPONSE_DENOMINATOR_SUPPORT_THRESHOLD,
) -> dict[str, float | int | bool]:
    """Diagnose spectral support for every channel used by the response ratios.

    The default requires each angle, frequency, and ROCOF coefficient to be at
    least twice its local background amplitude. This is an observable support
    rule, not a floating-point clipping convention.
    """

    prepared = dict(channels)
    prepared["voltage_angle_deg"] = unwrap_reported_angle_deg(
        channels["voltage_angle_deg"]
    )
    output: dict[str, float | int | bool] = {}
    support_channels = (
        ("voltage_angle_deg", "angle"),
        ("frequency_hz", "frequency"),
        ("rocof_hz_per_s", "rocof"),
    )
    for name, prefix in support_channels:
        details = spectral_support_details(
            time_s,
            prepared[name],
            observed_alias_hz,
            report_rate_hz,
            background_offsets_hz=background_offsets_hz,
        )
        for key, value in details.items():
            output[f"{prefix}_{key}"] = value
    output["minimum_required_log_support"] = float(minimum_log_support)
    output["support_pass"] = bool(
        all(
            float(output[f"{prefix}_log_support"]) >= float(minimum_log_support)
            for _, prefix in support_channels
        )
    )
    return output


def extract_complex_response_features(
    time_s: np.ndarray,
    channels: Mapping[str, np.ndarray],
    observed_alias_hz: float,
) -> dict[str, float]:
    """Return phase-wrap-safe features at one folded oscillation frequency."""

    missing = [name for name in REQUIRED_CHANNELS if name not in channels]
    if missing:
        raise ValueError(f"Missing PMU channels: {missing}")
    prepared_channels = dict(channels)
    prepared_channels["voltage_angle_deg"] = unwrap_reported_angle_deg(
        channels["voltage_angle_deg"]
    )
    coefficients = {
        name: complex_amplitude(time_s, prepared_channels[name], observed_alias_hz)
        for name in REQUIRED_CHANNELS
    }
    angle_coefficient = coefficients["voltage_angle_deg"]
    frequency_coefficient = coefficients["frequency_hz"]
    if abs(angle_coefficient) == 0.0 or abs(frequency_coefficient) == 0.0:
        raise ValueError("Response-ratio denominator coefficient is exactly zero")
    frequency_angle = (
        180.0
        / math.pi
        * frequency_coefficient
        / angle_coefficient
    )
    rocof_frequency = (
        coefficients["rocof_hz_per_s"]
        / (2.0 * math.pi)
        / frequency_coefficient
    )

    def phase_pair(value: complex) -> tuple[float, float]:
        magnitude = max(abs(value), RESPONSE_LOG_MAGNITUDE_FLOOR)
        return float(value.real / magnitude), float(value.imag / magnitude)

    frequency_cos, frequency_sin = phase_pair(frequency_angle)
    rocof_cos, rocof_sin = phase_pair(rocof_frequency)
    features = {
        "response_log_frequency_angle": float(
            math.log(max(abs(frequency_angle), RESPONSE_LOG_MAGNITUDE_FLOOR))
        ),
        "response_frequency_angle_cos": frequency_cos,
        "response_frequency_angle_sin": frequency_sin,
        "response_log_rocof_frequency": float(
            math.log(max(abs(rocof_frequency), RESPONSE_LOG_MAGNITUDE_FLOOR))
        ),
        "response_rocof_frequency_cos": rocof_cos,
        "response_rocof_frequency_sin": rocof_sin,
    }
    if not all(np.isfinite(value) for value in features.values()):
        raise ValueError("Complex response features must be finite")
    return features


@dataclass(frozen=True)
class VectorCandidateResponse:
    """Robust center, scales, and an enclosing response box for one candidate."""

    candidate_hz: float
    feature_names: tuple[str, ...]
    center: tuple[float, ...]
    scale: tuple[float, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    development_count: int
    calibration_count: int

    def nonconformity(self, observed: Mapping[str, float]) -> float:
        values = np.asarray([float(observed[name]) for name in self.feature_names])
        center = np.asarray(self.center)
        scale = np.asarray(self.scale)
        return float(np.sqrt(np.mean(((values - center) / scale) ** 2)))

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_hz": self.candidate_hz,
            "feature_names": list(self.feature_names),
            "center": list(self.center),
            "scale": list(self.scale),
            "lower": list(self.lower),
            "upper": list(self.upper),
            "development_count": self.development_count,
            "calibration_count": self.calibration_count,
        }


@dataclass(frozen=True)
class ComplexResponseModel:
    """Split-conformal vector response model for one declared measurement method."""

    estimator_configuration_id: str
    report_rate_hz: int
    feature_names: tuple[str, ...]
    target_miscoverage: float
    conformal_radius: float
    calibration_id: str
    response_by_candidate: Mapping[float, VectorCandidateResponse]
    calibration_sample_count: int
    certified: bool
    coverage_kind: str = (
        "marginal_response_score_conditional_on_correct_fold_and_exchangeability"
    )

    @property
    def box_projection_factor(self) -> float:
        """Coordinate projection factor for the RMS-standardized L2 region."""

        return math.sqrt(len(self.feature_names))

    @property
    def candidate_grid_hz(self) -> tuple[float, ...]:
        return tuple(sorted(float(value) for value in self.response_by_candidate))

    def response(self, candidate_hz: float) -> VectorCandidateResponse:
        key = min(
            self.response_by_candidate,
            key=lambda value: abs(float(value) - float(candidate_hz)),
        )
        if not math.isclose(float(key), float(candidate_hz), rel_tol=0.0, abs_tol=1e-9):
            raise KeyError(f"Candidate {candidate_hz:g} Hz is absent from response calibration")
        return self.response_by_candidate[key]

    def compatible(self, candidate_hz: float, observed: Mapping[str, float]) -> bool:
        return bool(
            self.certified
            and self.response(candidate_hz).nonconformity(observed)
            <= self.conformal_radius + 1e-12
        )

    def regions_overlap(self, left_hz: float, right_hz: float) -> bool:
        """Conservatively test overlap of conformal axis-aligned bounding boxes."""

        left = self.response(left_hz)
        right = self.response(right_hz)
        return bool(
            np.all(
                np.maximum(np.asarray(left.lower), np.asarray(right.lower))
                <= np.minimum(np.asarray(left.upper), np.asarray(right.upper)) + 1e-12
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "estimator_configuration_id": self.estimator_configuration_id,
            "report_rate_hz": self.report_rate_hz,
            "feature_names": list(self.feature_names),
            "target_miscoverage": self.target_miscoverage,
            "conformal_radius": self.conformal_radius,
            "calibration_id": self.calibration_id,
            "calibration_sample_count": self.calibration_sample_count,
            "certified": self.certified,
            "coverage_kind": self.coverage_kind,
            "response_region": {
                "nonconformity": "root_mean_square_standardized_l2",
                "bounding_box": "exact_axis_aligned_coordinate_projection",
                "box_projection_factor": self.box_projection_factor,
            },
            "response_by_candidate": {
                f"{candidate:g}": response.to_dict()
                for candidate, response in sorted(self.response_by_candidate.items())
            },
        }


def _rows_by_candidate(
    rows: Iterable[Mapping[str, object]], feature_names: Sequence[str]
) -> dict[float, list[np.ndarray]]:
    output: dict[float, list[np.ndarray]] = {}
    for row in rows:
        candidate = float(row["true_frequency_hz"])
        values = np.asarray([float(row[name]) for name in feature_names], dtype=np.float64)
        if not np.isfinite(candidate) or not np.all(np.isfinite(values)):
            raise ValueError("Response rows require finite frequency and feature values")
        output.setdefault(candidate, []).append(values)
    if not output:
        raise ValueError("At least one response row is required")
    return output


def fit_complex_response_model(
    development_rows: Iterable[Mapping[str, object]],
    calibration_rows: Iterable[Mapping[str, object]],
    *,
    estimator_configuration_id: str,
    report_rate_hz: int,
    feature_names: Sequence[str] = COMPLEX_RESPONSE_FEATURES,
    target_miscoverage: float = 0.1,
    minimum_scale: float = 0.02,
) -> ComplexResponseModel:
    """Fit robust vector responses and a pooled held-out conformal radius."""

    names = tuple(str(name) for name in feature_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("feature_names must be nonempty and unique")
    if minimum_scale <= 0:
        raise ValueError("minimum_scale must be positive")
    development = _rows_by_candidate(tuple(development_rows), names)
    calibration = _rows_by_candidate(tuple(calibration_rows), names)
    if set(development) != set(calibration):
        raise ValueError("Development and calibration candidate sets must match")

    centers: dict[float, np.ndarray] = {}
    scales: dict[float, np.ndarray] = {}
    scores: list[float] = []
    for candidate in sorted(development):
        matrix = np.vstack(development[candidate])
        center = np.median(matrix, axis=0)
        mad = np.median(np.abs(matrix - center), axis=0)
        scale = np.maximum(1.4826 * mad, float(minimum_scale))
        centers[candidate] = center
        scales[candidate] = scale
        for values in calibration[candidate]:
            scores.append(float(np.sqrt(np.mean(((values - center) / scale) ** 2))))

    radius, _ = conformal_order_statistic(np.asarray(scores), target_miscoverage)
    certified = bool(np.isfinite(radius))
    responses: dict[float, VectorCandidateResponse] = {}
    # The nonconformity uses the root-mean-square standardized L2 norm:
    # A(z)^2 = (1 / d) sum_j ((z_j - m_j) / s_j)^2.  Therefore A(z) <= q
    # implies |z_j - m_j| <= sqrt(d) q s_j for every coordinate.  The
    # sqrt(d) factor makes the intervals below the exact axis-aligned
    # coordinate projection of the calibrated ellipsoid.
    box_projection_factor = math.sqrt(len(names))
    for candidate in sorted(development):
        half_width = (
            box_projection_factor * radius * scales[candidate]
            if certified
            else np.full(len(names), np.inf)
        )
        responses[candidate] = VectorCandidateResponse(
            candidate_hz=candidate,
            feature_names=names,
            center=tuple(float(value) for value in centers[candidate]),
            scale=tuple(float(value) for value in scales[candidate]),
            lower=tuple(float(value) for value in centers[candidate] - half_width),
            upper=tuple(float(value) for value in centers[candidate] + half_width),
            development_count=len(development[candidate]),
            calibration_count=len(calibration[candidate]),
        )

    identity = {
        "estimator_configuration_id": estimator_configuration_id,
        "report_rate_hz": int(report_rate_hz),
        "feature_names": names,
        "target_miscoverage": float(target_miscoverage),
        "conformal_radius": float(radius),
        "box_projection_factor": box_projection_factor,
        "responses": {
            f"{candidate:g}": responses[candidate].to_dict()
            for candidate in sorted(responses)
        },
    }
    return ComplexResponseModel(
        estimator_configuration_id=estimator_configuration_id,
        report_rate_hz=int(report_rate_hz),
        feature_names=names,
        target_miscoverage=float(target_miscoverage),
        conformal_radius=float(radius),
        calibration_id="complex_response_" + stable_hash(identity)[:16],
        response_by_candidate=responses,
        calibration_sample_count=len(scores),
        certified=certified,
    )


def calibrated_vector_candidate_decision(
    observed_alias_hz: float,
    observed_features: Mapping[str, float],
    candidate_grid_hz: Sequence[float],
    report_rate_hz: float,
    *,
    response_model: ComplexResponseModel,
    estimator_configuration_id: str,
    response_calibration_id: str,
    event_score: float | None = None,
    require_disjoint_response_boxes: bool = False,
) -> dict[str, object]:
    """Resolve one already-detected alias or return a safe set/abstention."""

    if (
        not response_model.certified
        or response_model.report_rate_hz != int(report_rate_hz)
        or response_model.estimator_configuration_id != estimator_configuration_id
        or response_model.calibration_id != response_calibration_id
    ):
        return _decision_result(
            STATUS_ESTIMATOR_RESPONSE_NOT_CERTIFIED,
            observed_alias_hz=observed_alias_hz,
            event_score=event_score,
            response_calibration_id=response_calibration_id,
            reason="estimator, report rate, or calibration identifier is not certified",
        )
    if any(name not in observed_features for name in response_model.feature_names):
        return _decision_result(
            STATUS_ESTIMATOR_RESPONSE_NOT_CERTIFIED,
            observed_alias_hz=observed_alias_hz,
            event_score=event_score,
            response_calibration_id=response_calibration_id,
            reason="observed response feature schema does not match calibration",
        )
    grid = np.asarray(candidate_grid_hz, dtype=np.float64)
    if set(np.round(grid, 9)) != set(np.round(response_model.candidate_grid_hz, 9)):
        return _decision_result(
            STATUS_ESTIMATOR_RESPONSE_NOT_CERTIFIED,
            observed_alias_hz=observed_alias_hz,
            event_score=event_score,
            response_calibration_id=response_calibration_id,
            reason="candidate catalogue does not match response calibration",
        )
    alias_candidates = [
        float(candidate)
        for candidate in grid
        if math.isclose(
            alias_frequency(float(candidate), report_rate_hz),
            float(observed_alias_hz),
            abs_tol=1e-9,
        )
    ]
    if not alias_candidates:
        return _decision_result(
            STATUS_ESTIMATOR_RESPONSE_NOT_CERTIFIED,
            observed_alias_hz=observed_alias_hz,
            event_score=event_score,
            response_calibration_id=response_calibration_id,
            reason="no candidate is compatible with the detected alias",
        )
    if not is_resolvable_fold(alias_candidates[0], report_rate_hz):
        return _decision_result(
            STATUS_ALIAS_UNRESOLVED,
            candidates=alias_candidates,
            observed_alias_hz=observed_alias_hz,
            event_score=event_score,
            response_calibration_id=response_calibration_id,
            reason="zero or exact-Nyquist fold is not certified for a singleton",
        )
    if len(alias_candidates) == 1:
        return _decision_result(
            STATUS_PHYSICAL_FREQUENCY_SET,
            candidates=alias_candidates,
            observed_alias_hz=observed_alias_hz,
            event_score=event_score,
            response_calibration_id=response_calibration_id,
            reason="the reporting-rate fold uniquely identifies a catalogue candidate",
        )
    compatible = [
        candidate
        for candidate in alias_candidates
        if response_model.compatible(candidate, observed_features)
    ]
    if not compatible:
        return _decision_result(
            STATUS_ESTIMATOR_RESPONSE_NOT_CERTIFIED,
            candidates=alias_candidates,
            observed_alias_hz=observed_alias_hz,
            event_score=event_score,
            response_calibration_id=response_calibration_id,
            reason="observation lies outside every calibrated vector response",
        )
    if len(compatible) != 1:
        return _decision_result(
            STATUS_ALIAS_UNRESOLVED,
            candidates=compatible,
            observed_alias_hz=observed_alias_hz,
            event_score=event_score,
            response_calibration_id=response_calibration_id,
            reason="multiple alias-compatible vector responses contain the observation",
        )
    selected = compatible[0]
    if require_disjoint_response_boxes:
        overlapping = [
            candidate
            for candidate in alias_candidates
            if candidate != selected and response_model.regions_overlap(selected, candidate)
        ]
        if overlapping:
            return _decision_result(
                STATUS_ALIAS_UNRESOLVED,
                candidates=sorted([selected, *overlapping]),
                observed_alias_hz=observed_alias_hz,
                event_score=event_score,
                response_calibration_id=response_calibration_id,
                reason="calibrated vector response boxes overlap a competitor",
            )
    return _decision_result(
        STATUS_PHYSICAL_FREQUENCY_SET,
        candidates=[selected],
        observed_alias_hz=observed_alias_hz,
        event_score=event_score,
        response_calibration_id=response_calibration_id,
    )


def complex_calibrated_candidate_decision(
    time_s: np.ndarray,
    channels: Mapping[str, np.ndarray],
    candidate_grid_hz: Sequence[float],
    report_rate_hz: float,
    *,
    response_model: ComplexResponseModel,
    estimator_configuration_id: str,
    response_calibration_id: str,
    event_threshold: float = -math.inf,
    expected_frame_count: int | None = None,
    require_disjoint_response_boxes: bool = False,
    minimum_denominator_log_support: float = RESPONSE_DENOMINATOR_SUPPORT_THRESHOLD,
) -> dict[str, object]:
    """Detect a folded event and apply calibrated complex response sets."""

    refusal = deterministic_quality_gate(
        time_s,
        channels,
        report_rate_hz,
        expected_frame_count=expected_frame_count,
    )
    if refusal is not None:
        return _decision_result(refusal, reason="reported data failed deterministic quality checks")
    rows = scan_candidates(time_s, channels, candidate_grid_hz, report_rate_hz)
    alias_support: dict[float, float] = {}
    for row in rows:
        alias = float(row["observed_alias_hz"])
        alias_support[alias] = max(alias_support.get(alias, -math.inf), float(row["support"]))
    observed_alias_hz, event_score = select_max_support(alias_support)
    if event_score < float(event_threshold):
        return _decision_result(
            STATUS_NO_CATALOGUE_EVENT,
            event_score=float(event_score),
            observed_alias_hz=float(observed_alias_hz),
            response_calibration_id=response_model.calibration_id,
        )
    support = response_support_diagnostics(
        time_s,
        channels,
        observed_alias_hz,
        report_rate_hz,
        minimum_log_support=minimum_denominator_log_support,
    )
    if not bool(support["support_pass"]):
        return _decision_result(
            STATUS_INSUFFICIENT_RESPONSE_SUPPORT,
            candidates=[
                float(candidate)
                for candidate in candidate_grid_hz
                if math.isclose(
                    alias_frequency(float(candidate), report_rate_hz),
                    float(observed_alias_hz),
                    abs_tol=1e-9,
                )
            ],
            observed_alias_hz=float(observed_alias_hz),
            event_score=float(event_score),
            response_calibration_id=response_model.calibration_id,
            reason=(
                "angle, frequency, or ROCOF coefficient is below the declared "
                "local spectral-support threshold"
            ),
        )
    features = extract_complex_response_features(time_s, channels, observed_alias_hz)
    return calibrated_vector_candidate_decision(
        observed_alias_hz,
        features,
        candidate_grid_hz,
        report_rate_hz,
        response_model=response_model,
        estimator_configuration_id=estimator_configuration_id,
        response_calibration_id=response_calibration_id,
        event_score=float(event_score),
        require_disjoint_response_boxes=require_disjoint_response_boxes,
    )


__all__ = [
    "COMPLEX_RESPONSE_FEATURES",
    "RESPONSE_CHANNEL_SUPPORT_THRESHOLD",
    "RESPONSE_DENOMINATOR_SUPPORT_THRESHOLD",
    "ComplexResponseModel",
    "VectorCandidateResponse",
    "calibrated_vector_candidate_decision",
    "complex_calibrated_candidate_decision",
    "extract_complex_response_features",
    "fit_complex_response_model",
    "response_support_diagnostics",
]
