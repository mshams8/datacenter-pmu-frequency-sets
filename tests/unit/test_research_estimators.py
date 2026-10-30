from __future__ import annotations

import numpy as np
import pytest

from pmu_sso.config import SyntheticPMUChainConfig
from pmu_sso.inference.candidate_scan import (
    STATUS_ALIAS_UNRESOLVED,
    STATUS_ESTIMATOR_RESPONSE_NOT_CERTIFIED,
    STATUS_PHYSICAL_FREQUENCY_SET,
    calibrated_candidate_decision,
    scan_candidates,
)
from pmu_sso.response.estimator_response import fit_estimator_response_model
from pmu_sso.response.complex_response import (
    COMPLEX_RESPONSE_FEATURES,
    calibrated_vector_candidate_decision,
    extract_complex_response_features,
    fit_complex_response_model,
)
from pmu_sso.simulation.research_estimators import (
    estimate_research_pmu_channels,
    estimate_scalar_research_pmu_channels,
    resample_scalar_waveform_polyphase,
)


def phase_modulated_waveform(
    event_frequency_hz: float,
    *,
    sample_rate_hz: int = 4_800,
    nominal_frequency_hz: float = 60.0,
    duration_s: float = 3.0,
) -> np.ndarray:
    time_s = np.arange(int(sample_rate_hz * duration_s), dtype=float) / sample_rate_hz
    event_angle = np.deg2rad(0.10) * np.sin(2 * np.pi * event_frequency_hz * time_s + 0.3)
    magnitude = 1.0 + 0.002 * np.cos(2 * np.pi * event_frequency_hz * time_s - 0.2)
    carrier = np.exp(1j * (2 * np.pi * nominal_frequency_hz * time_s + event_angle))
    rotations = np.exp(1j * np.deg2rad([0.0, -120.0, 120.0]))
    phasors = magnitude[:, None] * carrier[:, None] * rotations[None, :]
    return np.sqrt(2.0) * np.real(phasors)


def chain(kind: str, stage: str, chain_id: str = "test") -> SyntheticPMUChainConfig:
    return SyntheticPMUChainConfig(
        chain_id=chain_id,
        estimator_kind=kind,
        frequency_stage=stage,
        implementation_version="unit_v1",
        front_end_filter_id="none",
        window_cycles=1,
        window_type="rectangular",
    )


def synthetic_reported_channels(
    true_frequency_hz: float,
    *,
    response_frequency_hz: float | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    report_rate_hz = 60.0
    time_s = np.arange(0.0, 3.0, 1.0 / report_rate_hz)
    angle_amplitude_deg = 0.02
    angle = angle_amplitude_deg * np.sin(2.0 * np.pi * true_frequency_hz * time_s)
    response_frequency = (
        true_frequency_hz if response_frequency_hz is None else response_frequency_hz
    )
    frequency = (
        2.0
        * np.pi
        * response_frequency
        / 360.0
        * angle_amplitude_deg
        * np.cos(2.0 * np.pi * true_frequency_hz * time_s)
    )
    return time_s, {
        "voltage_magnitude_pu": 1.0
        + 0.001 * np.sin(2.0 * np.pi * true_frequency_hz * time_s),
        "voltage_angle_deg": angle,
        "frequency_hz": 60.0 + frequency,
        "rocof_hz_per_s": np.gradient(frequency, 1.0 / report_rate_hz),
    }


def response_rows(center_by_frequency: dict[float, float], count: int = 12):
    rows = []
    offsets = np.linspace(-0.002, 0.002, count)
    for frequency, center in center_by_frequency.items():
        rows.extend(
            {
                "true_frequency_hz": frequency,
                "angle_ratio_hz": center * float(np.exp(offset)),
            }
            for offset in offsets
        )
    return rows


def test_estimator_configuration_rejects_invalid_frequency_stage() -> None:
    with pytest.raises(ValueError):
        chain("taylor_fourier", "reported_phase_derivative")


def test_estimator_configuration_scopes_taylor_settings() -> None:
    with pytest.raises(ValueError):
        SyntheticPMUChainConfig(
            chain_id="invalid",
            estimator_kind="sliding_dft",
            frequency_stage="reported_phase_derivative",
            window_cycles=1,
            window_type="rectangular",
            taylor_order=2,
        )
    configured = SyntheticPMUChainConfig(
        chain_id="tf2",
        estimator_kind="taylor_fourier",
        frequency_stage="taylor_phasor_derivative",
        window_cycles=2,
        window_type="hann",
        taylor_order=2,
        taylor_ridge_relative=1e-6,
    )
    assert configured.taylor_order == 2


def test_predecimation_and_reported_phase_expose_different_alias_response() -> None:
    waveform = phase_modulated_waveform(55.0)
    pre = estimate_research_pmu_channels(
        waveform,
        sample_rate_hz=4_800,
        nominal_frequency_hz=60.0,
        report_rate_hz=60,
        chain=chain("sliding_dft", "predecimation_phase_derivative", "pre"),
    )
    post = estimate_research_pmu_channels(
        waveform,
        sample_rate_hz=4_800,
        nominal_frequency_hz=60.0,
        report_rate_hz=60,
        chain=chain("sliding_dft", "reported_phase_derivative", "post"),
    )
    pre_row = next(
        row
        for row in scan_candidates(pre.timestamp_s, pre.channels, [5.0, 55.0], 60.0)
        if row["candidate_hz"] == 55.0
    )
    post_row = next(
        row
        for row in scan_candidates(post.timestamp_s, post.channels, [5.0, 55.0], 60.0)
        if row["candidate_hz"] == 55.0
    )
    assert pre_row["angle_ratio_hz"] == pytest.approx(55.0, rel=0.08)
    assert post_row["angle_ratio_hz"] == pytest.approx(5.0, rel=0.15)


def test_taylor_fourier_estimator_returns_finite_reported_channels() -> None:
    estimate = estimate_research_pmu_channels(
        phase_modulated_waveform(25.0),
        sample_rate_hz=4_800,
        nominal_frequency_hz=60.0,
        report_rate_hz=60,
        chain=chain("taylor_fourier", "taylor_phasor_derivative", "tf"),
    )
    assert len(estimate.timestamp_s) > 100
    assert all(np.all(np.isfinite(values)) for values in estimate.channels.values())


def test_higher_order_taylor_fourier_uses_global_reference_at_48_fps() -> None:
    configured = SyntheticPMUChainConfig(
        chain_id="tf3",
        estimator_kind="taylor_fourier",
        frequency_stage="taylor_phasor_derivative",
        implementation_version="unit_v2",
        front_end_filter_id="none",
        window_cycles=2,
        window_type="hann",
        taylor_order=3,
        taylor_ridge_relative=1e-6,
    )
    estimate = estimate_research_pmu_channels(
        phase_modulated_waveform(5.0),
        sample_rate_hz=4_800,
        nominal_frequency_hz=60.0,
        report_rate_hz=48,
        chain=configured,
    )
    assert np.median(np.abs(estimate.channels["frequency_hz"] - 60.0)) < 0.02


def vector_response_rows(center_by_frequency: dict[float, float], count: int = 12):
    rows = []
    offsets = np.linspace(-0.002, 0.002, count)
    for frequency, center in center_by_frequency.items():
        for offset in offsets:
            row = {"true_frequency_hz": frequency}
            row.update(
                {
                    name: center + float(offset) + 0.05 * index
                    for index, name in enumerate(COMPLEX_RESPONSE_FEATURES)
                }
            )
            rows.append(row)
    return rows


def test_complex_features_are_finite_and_phase_wrap_safe() -> None:
    time_s, channels = synthetic_reported_channels(7.0)
    features = extract_complex_response_features(time_s, channels, 7.0)
    assert set(features) == set(COMPLEX_RESPONSE_FEATURES)
    assert all(np.isfinite(value) for value in features.values())
    assert (
        features["response_frequency_angle_cos"] ** 2
        + features["response_frequency_angle_sin"] ** 2
    ) == pytest.approx(1.0)


def test_complex_features_unwrap_reported_angle_before_spectral_analysis() -> None:
    time_s, channels = synthetic_reported_channels(7.0)
    unwrapped = dict(channels)
    unwrapped["voltage_angle_deg"] = 179.95 + 0.10 * np.sin(
        2.0 * np.pi * 7.0 * time_s
    )
    wrapped = dict(unwrapped)
    wrapped["voltage_angle_deg"] = (
        unwrapped["voltage_angle_deg"] + 180.0
    ) % 360.0 - 180.0

    reference = extract_complex_response_features(time_s, unwrapped, 7.0)
    observed = extract_complex_response_features(time_s, wrapped, 7.0)
    for name in COMPLEX_RESPONSE_FEATURES:
        assert observed[name] == pytest.approx(reference[name], abs=1e-10)


def test_vector_response_selects_separated_candidate_and_rejects_metadata_mismatch() -> None:
    rows = vector_response_rows({7.0: 0.0, 53.0: 2.0})
    model = fit_complex_response_model(
        rows,
        rows,
        estimator_configuration_id="vector_60fps",
        report_rate_hz=60,
    )
    observed = {
        name: 2.0 + 0.05 * index
        for index, name in enumerate(COMPLEX_RESPONSE_FEATURES)
    }
    selected = calibrated_vector_candidate_decision(
        7.0,
        observed,
        [7.0, 53.0],
        60.0,
        response_model=model,
        estimator_configuration_id="vector_60fps",
        response_calibration_id=model.calibration_id,
    )
    assert selected["status"] == STATUS_PHYSICAL_FREQUENCY_SET
    assert selected["candidate_hz"] == [53.0]
    mismatch = calibrated_vector_candidate_decision(
        7.0,
        observed,
        [7.0, 53.0],
        60.0,
        response_model=model,
        estimator_configuration_id="wrong",
        response_calibration_id=model.calibration_id,
    )
    assert mismatch["status"] == STATUS_ESTIMATOR_RESPONSE_NOT_CERTIFIED


def test_unique_fold_does_not_require_response_compatibility() -> None:
    rows = vector_response_rows({7.0: 0.0, 53.0: 2.0})
    model = fit_complex_response_model(
        rows,
        rows,
        estimator_configuration_id="vector_120fps",
        report_rate_hz=120,
    )
    far_outside = {name: 1e6 for name in COMPLEX_RESPONSE_FEATURES}
    selected = calibrated_vector_candidate_decision(
        53.0,
        far_outside,
        [7.0, 53.0],
        120.0,
        response_model=model,
        estimator_configuration_id="vector_120fps",
        response_calibration_id=model.calibration_id,
    )
    assert selected["status"] == STATUS_PHYSICAL_FREQUENCY_SET
    assert selected["candidate_hz"] == [53.0]


def test_vector_response_box_encloses_rms_standardized_conformal_region() -> None:
    rows = vector_response_rows({7.0: 0.0, 53.0: 2.0})
    model = fit_complex_response_model(
        rows,
        rows,
        estimator_configuration_id="vector_box_geometry_60fps",
        report_rate_hz=60,
    )
    response = model.response(7.0)
    center = np.asarray(response.center)
    scale = np.asarray(response.scale)
    lower = np.asarray(response.lower)
    upper = np.asarray(response.upper)
    dimension = len(response.feature_names)

    expected_half_width = np.sqrt(dimension) * model.conformal_radius * scale
    np.testing.assert_allclose(center - lower, expected_half_width)
    np.testing.assert_allclose(upper - center, expected_half_width)
    assert model.box_projection_factor == pytest.approx(np.sqrt(dimension))

    rng = np.random.default_rng(20260830)
    for _ in range(200):
        direction = rng.normal(size=dimension)
        direction /= np.linalg.norm(direction)
        standardized = (
            direction
            * np.sqrt(dimension)
            * model.conformal_radius
            * rng.uniform(0.0, 1.0)
        )
        point = center + scale * standardized
        observed = dict(zip(response.feature_names, point, strict=True))
        assert response.nonconformity(observed) <= model.conformal_radius + 1e-12
        assert np.all(point >= lower - 1e-12)
        assert np.all(point <= upper + 1e-12)


def test_calibrated_decision_selects_separated_response() -> None:
    time_s, channels = synthetic_reported_channels(53.0)
    estimator_id = "unit_estimator_60fps"
    model = fit_estimator_response_model(
        response_rows({7.0: 7.0, 53.0: 53.0}),
        response_rows({7.0: 7.0, 53.0: 53.0}),
        estimator_configuration_id=estimator_id,
        report_rate_hz=60,
    )
    result = calibrated_candidate_decision(
        time_s,
        channels,
        [7.0, 53.0],
        60.0,
        response_model=model,
        estimator_configuration_id=estimator_id,
        response_calibration_id=model.calibration_id,
    )
    assert result["status"] == STATUS_PHYSICAL_FREQUENCY_SET
    assert result["candidate_hz"] == [53.0]


def test_calibrated_decision_abstains_for_overlapping_response() -> None:
    time_s, channels = synthetic_reported_channels(53.0, response_frequency_hz=7.0)
    estimator_id = "reported_phase_60fps"
    model = fit_estimator_response_model(
        response_rows({7.0: 7.0, 53.0: 7.0}),
        response_rows({7.0: 7.0, 53.0: 7.0}),
        estimator_configuration_id=estimator_id,
        report_rate_hz=60,
    )
    result = calibrated_candidate_decision(
        time_s,
        channels,
        [7.0, 53.0],
        60.0,
        response_model=model,
        estimator_configuration_id=estimator_id,
        response_calibration_id=model.calibration_id,
    )
    assert result["status"] == STATUS_ALIAS_UNRESOLVED
    assert result["candidate_hz"] == [7.0, 53.0]


def test_calibrated_decision_rejects_metadata_mismatch() -> None:
    time_s, channels = synthetic_reported_channels(53.0)
    model = fit_estimator_response_model(
        response_rows({7.0: 7.0, 53.0: 53.0}),
        response_rows({7.0: 7.0, 53.0: 53.0}),
        estimator_configuration_id="expected",
        report_rate_hz=60,
    )
    result = calibrated_candidate_decision(
        time_s,
        channels,
        [7.0, 53.0],
        60.0,
        response_model=model,
        estimator_configuration_id="mismatch",
        response_calibration_id=model.calibration_id,
    )
    assert result["status"] == STATUS_ESTIMATOR_RESPONSE_NOT_CERTIFIED


def scalar_carrier_with_sso(
    sso_frequency_hz: float | None = None,
    *,
    sample_rate_hz: int = 6_000,
    nominal_frequency_hz: float = 50.0,
    duration_s: float = 5.0,
    carrier_phase_rad: float = 0.4,
) -> np.ndarray:
    time_s = np.arange(int(sample_rate_hz * duration_s), dtype=float) / sample_rate_hz
    waveform = np.sqrt(2.0) * np.cos(
        2.0 * np.pi * nominal_frequency_hz * time_s + carrier_phase_rad
    )
    if sso_frequency_hz is not None:
        waveform += 0.08 * np.sqrt(2.0) * np.cos(
            2.0 * np.pi * sso_frequency_hz * time_s - 0.2
        )
    return waveform


def dominant_complex_frequency_hz(
    time_s: np.ndarray, values: np.ndarray, low_hz: float, high_hz: float
) -> float:
    interval_s = float(np.median(np.diff(time_s)))
    centered = values - np.mean(values)
    spectrum = np.fft.fft(centered * np.hanning(len(centered)))
    frequency = np.fft.fftfreq(len(centered), d=interval_s)
    mask = (np.abs(frequency) >= low_hz) & (np.abs(frequency) <= high_hz)
    return float(abs(frequency[mask][np.argmax(np.abs(spectrum[mask]))]))


def test_scalar_50_hz_carrier_has_registered_phase_and_frequency() -> None:
    estimate = estimate_scalar_research_pmu_channels(
        scalar_carrier_with_sso(),
        sample_rate_hz=6_000,
        nominal_frequency_hz=50.0,
        report_rate_hz=100,
        chain=chain("sliding_dft", "reported_phase_derivative", "scalar_phase"),
        signal_kind="current",
    )
    interior = slice(5, -5)
    assert np.median(np.abs(estimate.complex_phasor[interior])) == pytest.approx(
        1.0, rel=2e-3
    )
    phase_error = np.angle(
        np.exp(1j * (np.angle(estimate.complex_phasor[interior]) - 0.4))
    )
    assert np.max(np.abs(phase_error)) < 2e-3
    assert np.median(estimate.channels["frequency_hz"][interior]) == pytest.approx(
        50.0, abs=2e-3
    )


@pytest.mark.parametrize(
    ("report_rate_hz", "expected_fold_hz"), [(30, 11.7), (60, 18.3), (100, 41.7), (120, 41.7)]
)
def test_scalar_sso_replay_exposes_expected_reporting_rate_fold(
    report_rate_hz: int, expected_fold_hz: float
) -> None:
    estimate = estimate_scalar_research_pmu_channels(
        scalar_carrier_with_sso(8.3),
        sample_rate_hz=6_000,
        nominal_frequency_hz=50.0,
        report_rate_hz=report_rate_hz,
        chain=chain("sliding_dft", "reported_phase_derivative", "scalar_fold"),
        signal_kind="current",
    )
    frequency_hz = dominant_complex_frequency_hz(
        estimate.timestamp_s[3:-3], estimate.complex_phasor[3:-3], 1.0, report_rate_hz / 2
    )
    assert frequency_hz == pytest.approx(expected_fold_hz, abs=0.25)
    assert np.median(np.diff(estimate.timestamp_s)) == pytest.approx(1 / report_rate_hz)


def test_scalar_taylor_fourier_returns_finite_generic_channels() -> None:
    configured = SyntheticPMUChainConfig(
        chain_id="scalar_tf",
        estimator_kind="taylor_fourier",
        frequency_stage="taylor_phasor_derivative",
        implementation_version="unit_v1",
        front_end_filter_id="none",
        window_cycles=2,
        window_type="hann",
        taylor_order=2,
        taylor_ridge_relative=1e-6,
    )
    estimate = estimate_scalar_research_pmu_channels(
        scalar_carrier_with_sso(8.3),
        sample_rate_hz=6_000,
        nominal_frequency_hz=50.0,
        report_rate_hz=120,
        chain=configured,
        signal_kind="current",
    )
    assert set(estimate.channels) == {
        "phasor_magnitude_pu",
        "phasor_angle_deg",
        "frequency_hz",
        "rocof_hz_per_s",
    }
    assert all(np.all(np.isfinite(values)) for values in estimate.channels.values())


def test_polyphase_resampling_is_deterministic_and_time_registered() -> None:
    source = scalar_carrier_with_sso(8.3, sample_rate_hz=1_000, duration_s=1.0)
    first = resample_scalar_waveform_polyphase(
        source, input_sample_rate_hz=1_000, output_sample_rate_hz=6_000
    )
    second = resample_scalar_waveform_polyphase(
        source, input_sample_rate_hz=1_000, output_sample_rate_hz=6_000
    )
    assert first.up == 6 and first.down == 1
    assert len(first.waveform) == 6 * len(source)
    np.testing.assert_array_equal(first.waveform, second.waveform)
    np.testing.assert_array_equal(first.timestamp_s, second.timestamp_s)
    assert first.timestamp_s[0] == 0.0
    assert np.median(np.diff(first.timestamp_s)) == pytest.approx(1 / 6_000)
