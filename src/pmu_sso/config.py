"""Unchanged chain definition extracted from the study configuration."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel,ConfigDict,Field,model_validator

class SyntheticPMUChainConfig(BaseModel):
    """Transparent reference phasor-estimation chain."""

    model_config = ConfigDict(extra="forbid")

    chain_id: str
    estimator_kind: Literal["sliding_dft", "taylor_fourier"] = "sliding_dft"
    frequency_stage: Literal[
        "predecimation_phase_derivative",
        "reported_phase_derivative",
        "taylor_phasor_derivative",
    ] = "predecimation_phase_derivative"
    implementation_version: str = "research_v1"
    front_end_filter_id: str = "none"
    window_cycles: int = Field(ge=1, le=4)
    window_type: Literal["rectangular", "hann"]
    taylor_order: int = Field(default=1, ge=1, le=3)
    taylor_ridge_relative: float = Field(default=0.0, ge=0.0, le=0.1)
    timestamp_jitter_std_s: float = Field(default=0.0, ge=0)
    packet_loss_probability: float = Field(default=0.0, ge=0, lt=0.5)
    quantization_relative_step: float = Field(default=0.0, ge=0, le=0.01)

    @model_validator(mode="after")
    def validate_estimator_frequency_stage(self) -> "SyntheticPMUChainConfig":
        if (
            self.estimator_kind == "taylor_fourier"
            and self.frequency_stage != "taylor_phasor_derivative"
        ):
            raise ValueError(
                "taylor_fourier requires frequency_stage='taylor_phasor_derivative'"
            )
        if (
            self.estimator_kind == "sliding_dft"
            and self.frequency_stage == "taylor_phasor_derivative"
        ):
            raise ValueError(
                "sliding_dft does not support frequency_stage='taylor_phasor_derivative'"
            )
        if self.estimator_kind != "taylor_fourier" and (
            self.taylor_order != 1 or self.taylor_ridge_relative != 0.0
        ):
            raise ValueError(
                "Taylor order and ridge settings apply only to taylor_fourier estimators"
            )
        return self
