"""Estimator-specific response calibration for aliased-frequency candidates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from ..inference.calibration import conformal_order_statistic
from ..provenance import stable_hash


@dataclass(frozen=True)
class CandidateResponse:
    """Log-scale response location and uncertainty for one physical frequency."""

    candidate_hz: float
    center_ratio_hz: float
    log_scale: float
    lower_ratio_hz: float
    upper_ratio_hz: float
    development_count: int
    calibration_count: int

    def nonconformity(self, observed_ratio_hz: float) -> float:
        observed = max(float(observed_ratio_hz), np.finfo(float).tiny)
        return float(
            abs(math.log(observed / self.center_ratio_hz)) / max(self.log_scale, 1e-12)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_hz": self.candidate_hz,
            "center_ratio_hz": self.center_ratio_hz,
            "log_scale": self.log_scale,
            "lower_ratio_hz": self.lower_ratio_hz,
            "upper_ratio_hz": self.upper_ratio_hz,
            "development_count": self.development_count,
            "calibration_count": self.calibration_count,
        }


@dataclass(frozen=True)
class EstimatorResponseModel:
    """Calibrated response model for one estimator configuration and report rate."""

    estimator_configuration_id: str
    report_rate_hz: int
    target_miscoverage: float
    conformal_radius: float
    calibration_id: str
    response_by_candidate: Mapping[float, CandidateResponse]
    calibration_sample_count: int
    certified: bool
    coverage_kind: str = (
        "marginal_response_score_conditional_on_correct_fold_and_exchangeability"
    )

    @property
    def candidate_grid_hz(self) -> tuple[float, ...]:
        return tuple(sorted(float(value) for value in self.response_by_candidate))

    def response(self, candidate_hz: float) -> CandidateResponse:
        key = min(
            self.response_by_candidate,
            key=lambda value: abs(float(value) - float(candidate_hz)),
        )
        if not math.isclose(float(key), float(candidate_hz), rel_tol=0.0, abs_tol=1e-9):
            raise KeyError(f"Candidate {candidate_hz:g} Hz is absent from response calibration")
        return self.response_by_candidate[key]

    def nonconformity(self, candidate_hz: float, observed_ratio_hz: float) -> float:
        return self.response(candidate_hz).nonconformity(observed_ratio_hz)

    def compatible(self, candidate_hz: float, observed_ratio_hz: float) -> bool:
        return bool(
            self.certified
            and self.nonconformity(candidate_hz, observed_ratio_hz)
            <= self.conformal_radius + 1e-12
        )

    def intervals_overlap(self, left_hz: float, right_hz: float) -> bool:
        left = self.response(left_hz)
        right = self.response(right_hz)
        return bool(
            max(left.lower_ratio_hz, right.lower_ratio_hz)
            <= min(left.upper_ratio_hz, right.upper_ratio_hz) + 1e-12
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "estimator_configuration_id": self.estimator_configuration_id,
            "report_rate_hz": self.report_rate_hz,
            "target_miscoverage": self.target_miscoverage,
            "conformal_radius": self.conformal_radius,
            "calibration_id": self.calibration_id,
            "calibration_sample_count": self.calibration_sample_count,
            "certified": self.certified,
            "coverage_kind": self.coverage_kind,
            "response_by_candidate": {
                f"{candidate:g}": response.to_dict()
                for candidate, response in sorted(self.response_by_candidate.items())
            },
        }


def _rows_by_candidate(
    rows: Iterable[Mapping[str, object]],
) -> dict[float, list[float]]:
    output: dict[float, list[float]] = {}
    for row in rows:
        candidate = float(row["true_frequency_hz"])
        ratio = float(row["angle_ratio_hz"])
        if not np.isfinite(candidate) or not np.isfinite(ratio) or ratio <= 0:
            raise ValueError("Response rows require finite positive frequency and ratio")
        output.setdefault(candidate, []).append(ratio)
    if not output:
        raise ValueError("At least one response row is required")
    return output


def fit_estimator_response_model(
    development_rows: Iterable[Mapping[str, object]],
    calibration_rows: Iterable[Mapping[str, object]],
    *,
    estimator_configuration_id: str,
    report_rate_hz: int,
    target_miscoverage: float = 0.1,
    minimum_log_scale: float = 1e-3,
) -> EstimatorResponseModel:
    """Fit development centers/scales and a held-out conformal radius.

    The calibration score is pooled over true candidates.  The resulting
    guarantee is therefore marginal true-candidate coverage under the stated
    split/exchangeability assumptions; it is not a simultaneous guarantee over
    every scanned candidate.
    """

    if minimum_log_scale <= 0:
        raise ValueError("minimum_log_scale must be positive")
    development = _rows_by_candidate(tuple(development_rows))
    calibration = _rows_by_candidate(tuple(calibration_rows))
    if set(development) != set(calibration):
        raise ValueError("Development and calibration candidate sets must match")

    centers: dict[float, float] = {}
    scales: dict[float, float] = {}
    calibration_scores: list[float] = []
    for candidate in sorted(development):
        development_log = np.log(np.asarray(development[candidate], dtype=np.float64))
        center_log = float(np.median(development_log))
        mad = float(np.median(np.abs(development_log - center_log)))
        scale = max(1.4826 * mad, float(minimum_log_scale))
        center = float(math.exp(center_log))
        centers[candidate] = center
        scales[candidate] = scale
        calibration_scores.extend(
            abs(math.log(value / center)) / scale for value in calibration[candidate]
        )

    radius, _ = conformal_order_statistic(
        np.asarray(calibration_scores, dtype=np.float64), target_miscoverage
    )
    certified = bool(np.isfinite(radius))
    responses: dict[float, CandidateResponse] = {}
    for candidate in sorted(development):
        half_width = float(radius * scales[candidate]) if certified else math.inf
        lower = float(centers[candidate] * math.exp(-half_width)) if certified else 0.0
        upper = float(centers[candidate] * math.exp(half_width)) if certified else math.inf
        responses[candidate] = CandidateResponse(
            candidate_hz=candidate,
            center_ratio_hz=centers[candidate],
            log_scale=scales[candidate],
            lower_ratio_hz=lower,
            upper_ratio_hz=upper,
            development_count=len(development[candidate]),
            calibration_count=len(calibration[candidate]),
        )

    identity_payload = {
        "estimator_configuration_id": estimator_configuration_id,
        "report_rate_hz": int(report_rate_hz),
        "target_miscoverage": float(target_miscoverage),
        "conformal_radius": radius,
        "responses": {
            f"{candidate:g}": {
                "center": centers[candidate],
                "scale": scales[candidate],
                "development_count": len(development[candidate]),
                "calibration_count": len(calibration[candidate]),
            }
            for candidate in sorted(development)
        },
    }
    calibration_id = "estimator_response_" + stable_hash(identity_payload)[:16]
    return EstimatorResponseModel(
        estimator_configuration_id=estimator_configuration_id,
        report_rate_hz=int(report_rate_hz),
        target_miscoverage=float(target_miscoverage),
        conformal_radius=float(radius),
        calibration_id=calibration_id,
        response_by_candidate=responses,
        calibration_sample_count=len(calibration_scores),
        certified=certified,
    )


__all__ = [
    "CandidateResponse",
    "EstimatorResponseModel",
    "fit_estimator_response_model",
]
