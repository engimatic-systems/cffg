"""Basal-only simglucose experiment for unannounced meals.

This module is intentionally an experiment artifact, not a core benchmark:
the protocol is Cooper's 8-hour adults+adolescents challenge, while
``cffg.benchmarks.glucose`` preserves the older 24-hour adults-only reference.

The controller is white-box and basal-only. It uses a small discrete action set
of basal multipliers, a clock prior over likely meal times, CGM trend feedback,
and a crude insulin-on-board state. Meals are not passed to the controller.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from importlib import resources
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from cffg.benchmarks.glucose import _shim_legacy_gym

_shim_legacy_gym()

from simglucose.actuator.pump import InsulinPump  # noqa: E402
from simglucose.controller.base import Action  # noqa: E402
from simglucose.patient.t1dpatient import T1DPatient  # noqa: E402
from simglucose.sensor.cgm import CGMSensor  # noqa: E402
from simglucose.simulation.env import T1DSimEnv  # noqa: E402
from simglucose.simulation.scenario import CustomScenario  # noqa: E402

ADULTS = [f"adult#{i:03d}" for i in range(1, 11)]
ADOLESCENTS = [f"adolescent#{i:03d}" for i in range(1, 11)]
COHORT = [*ADULTS, *ADOLESCENTS]

RANGE_LO = 70.0
RANGE_HI = 180.0
CRITICAL_LO = 54.0
CRITICAL_HI = 250.0


@dataclass(frozen=True)
class Protocol:
    """Eight virtual hours with relative meals inside the episode."""

    start: datetime = datetime(2018, 1, 1, 6, 0)
    hours: float = 8.0
    meals: tuple[tuple[float, float], ...] = ((1.0, 45.0), (5.0, 70.0))
    seed: int = 7


@dataclass(frozen=True)
class TrialSpec:
    trial: int
    patient: str
    protocol: Protocol


@dataclass(frozen=True)
class MealPriorWindow:
    center_min: float
    pre_min: float
    post_min: float
    multiplier: float

    def active(self, elapsed_min: float) -> bool:
        return self.center_min - self.pre_min <= elapsed_min <= self.center_min + self.post_min


@dataclass(frozen=True)
class BasalPolicyConfig:
    """Discrete temp-basal controller settings.

    The windows encode a prior over the hidden meal schedule, not announced meal
    observations. The policy can only emit basal insulin; bolus is always zero.
    """

    basal_multipliers: tuple[float, ...] = (
        0.0,
        0.25,
        0.5,
        1.0,
        2.0,
        4.0,
        8.0,
        12.0,
        18.0,
        25.0,
        30.0,
    )
    windows: tuple[MealPriorWindow, ...] = (
        MealPriorWindow(center_min=60.0, pre_min=75.0, post_min=120.0, multiplier=18.0),
        MealPriorWindow(center_min=300.0, pre_min=75.0, post_min=120.0, multiplier=30.0),
    )
    suspend_cgm: float = 105.0
    predicted_low: float = 100.0
    falling_cgm: float = 130.0
    falling_trend: float = -0.3
    target_cgm: float = 110.0
    high_gain: float = 0.08
    trend_gain: float = 2.0
    meal_trend: float = 0.4
    meal_cgm: float = 105.0
    meal_multiplier: float = 25.0
    meal_trend_gain: float = 6.0
    down_cgm: float = 150.0
    down_multiplier: float = 0.2
    iob_gain: float = 0.03
    iob_half_life_min: float = 100.0
    forecast_min: float = 30.0


@dataclass
class PatientTrace:
    name: str
    trial: int | None = None
    patient: str | None = None
    meals: tuple[tuple[float, float], ...] = ()
    time_min: list[float] = field(default_factory=list)
    bg: list[float] = field(default_factory=list)
    cgm: list[float] = field(default_factory=list)
    basal: list[float] = field(default_factory=list)
    bolus: list[float] = field(default_factory=list)

    @property
    def tir(self) -> float:
        return _fraction(self.bg, lambda g: RANGE_LO <= g <= RANGE_HI)

    @property
    def time_low(self) -> float:
        return _fraction(self.bg, lambda g: g < RANGE_LO)

    @property
    def time_high(self) -> float:
        return _fraction(self.bg, lambda g: g > RANGE_HI)

    @property
    def critical_time(self) -> float:
        return _fraction(self.bg, lambda g: g < CRITICAL_LO or g > CRITICAL_HI)

    @property
    def critical_samples(self) -> int:
        return sum(1 for g in self.bg if g < CRITICAL_LO or g > CRITICAL_HI)

    @property
    def min_bg(self) -> float:
        return min(self.bg)

    @property
    def max_bg(self) -> float:
        return max(self.bg)


@dataclass(frozen=True)
class CohortTrace:
    patients: tuple[PatientTrace, ...]

    @property
    def tir(self) -> float:
        return float(np.mean([p.tir for p in self.patients]))

    @property
    def time_low(self) -> float:
        return float(np.mean([p.time_low for p in self.patients]))

    @property
    def time_high(self) -> float:
        return float(np.mean([p.time_high for p in self.patients]))

    @property
    def critical_time(self) -> float:
        return float(np.mean([p.critical_time for p in self.patients]))

    @property
    def critical_samples(self) -> int:
        return sum(p.critical_samples for p in self.patients)

    def summary(self) -> str:
        rows = [
            f"  {p.name}: TIR {p.tir:6.1%} low {p.time_low:5.1%} "
            f"high {p.time_high:5.1%} crit {p.critical_time:5.1%} "
            f"BG [{p.min_bg:5.1f}, {p.max_bg:5.1f}]"
            for p in self.patients
        ]
        head = (
            f"cohort TIR {self.tir:.2%} low {self.time_low:.2%} "
            f"high {self.time_high:.2%} crit {self.critical_time:.2%} "
            f"critical_samples {self.critical_samples}"
        )
        return "\n".join([head, *rows])


class ClockPriorBasalController:
    """CGM-only basal controller with a hidden-meal clock prior."""

    def __init__(
        self,
        patient_name: str,
        sample_min: float,
        config: BasalPolicyConfig = BasalPolicyConfig(),
    ):
        self.basal = _patient_basal(patient_name)
        self.sample_min = float(sample_min)
        self.config = config
        self.elapsed_min = 0.0
        self.previous_cgm: float | None = None
        self.extra_iob = 0.0
        self.iob_decay = 0.5 ** (self.sample_min / config.iob_half_life_min)

    def policy(self, cgm: float) -> Action:
        cfg = self.config
        trend = 0.0 if self.previous_cgm is None else (cgm - self.previous_cgm) / self.sample_min
        self.previous_cgm = cgm
        self.extra_iob *= self.iob_decay

        predicted = cgm + cfg.forecast_min * trend
        if (
            cgm < cfg.suspend_cgm
            or predicted < cfg.predicted_low
            or (cgm < cfg.falling_cgm and trend < cfg.falling_trend)
        ):
            multiplier = 0.0
        else:
            multiplier = 1.0
            for window in cfg.windows:
                if window.active(self.elapsed_min):
                    multiplier = max(multiplier, window.multiplier)

            high_feedback = 1.0 + cfg.high_gain * max(0.0, cgm - cfg.target_cgm)
            trend_feedback = cfg.trend_gain * max(0.0, trend)
            multiplier = max(multiplier, high_feedback + trend_feedback)

            if trend > cfg.meal_trend and cgm > cfg.meal_cgm:
                meal_feedback = cfg.meal_multiplier + cfg.meal_trend_gain * (trend - cfg.meal_trend)
                multiplier = max(multiplier, meal_feedback)

            if cgm < cfg.down_cgm and trend < 0.0:
                multiplier = min(multiplier, cfg.down_multiplier)

            multiplier -= cfg.iob_gain * self.extra_iob / max(self.basal, 1e-9)
            multiplier = _nearest_action(max(0.0, multiplier), cfg.basal_multipliers)

        basal = self.basal * multiplier
        self.extra_iob += max(0.0, basal - self.basal) * self.sample_min
        self.elapsed_min += self.sample_min
        return Action(basal=basal, bolus=0.0)


class ReactivePhysiologyBasalController:
    """Basal-only controller with meal-state inference from CGM dynamics.

    The physiological sketch is deliberately small: CGM velocity and
    acceleration raise a hidden meal-absorption drive; delivered temp basal
    raises an insulin-on-board state; both decay on separate time scales. The
    policy acts on predicted glucose and inferred meal drive, not known meal
    times.
    """

    def __init__(
        self,
        patient_name: str,
        sample_min: float,
        config: BasalPolicyConfig = BasalPolicyConfig(),
    ):
        self.basal = _patient_basal(patient_name)
        self.sample_min = float(sample_min)
        self.config = config
        self.previous_cgm: float | None = None
        self.previous_trend = 0.0
        self.meal_drive = 0.0
        self.extra_iob = 0.0
        self.meal_decay = 0.5 ** (self.sample_min / 55.0)
        self.iob_decay = 0.5 ** (self.sample_min / 100.0)

    def policy(self, cgm: float) -> Action:
        cfg = self.config
        trend = 0.0 if self.previous_cgm is None else (cgm - self.previous_cgm) / self.sample_min
        accel = trend - self.previous_trend
        self.previous_cgm = cgm
        self.previous_trend = trend

        self.meal_drive *= self.meal_decay
        self.extra_iob *= self.iob_decay
        if cgm > 105.0 and trend > 0.35:
            self.meal_drive += 10.0 * trend + 14.0 * max(0.0, accel)
        if cgm > 150.0 and trend > -0.1:
            self.meal_drive += 0.08 * (cgm - 150.0)
        self.meal_drive = min(self.meal_drive, 70.0)

        predicted = cgm + cfg.forecast_min * trend
        if cgm < 100.0 or predicted < 92.0 or (cgm < 125.0 and trend < -0.35):
            multiplier = 0.0
        else:
            multiplier = 1.0
            multiplier += 0.055 * max(0.0, cgm - 125.0)
            multiplier += 1.8 * max(0.0, trend)
            multiplier += 0.33 * self.meal_drive
            multiplier -= 0.035 * self.extra_iob / max(self.basal, 1e-9)
            if cgm < 145.0 and trend < 0.0:
                multiplier = min(multiplier, 0.5)
            multiplier = _nearest_action(
                max(0.0, multiplier),
                (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 18.0, 25.0, 30.0),
            )

        basal = self.basal * multiplier
        self.extra_iob += max(0.0, basal - self.basal) * self.sample_min
        return Action(basal=basal, bolus=0.0)


def run_patient(
    patient_name: str,
    *,
    protocol: Protocol = Protocol(),
    config: BasalPolicyConfig = BasalPolicyConfig(),
    controller: str = "clock",
    trial: int | None = None,
) -> PatientTrace:
    env = _build_env(patient_name, protocol)
    step = env.reset()
    basal_controller = _build_controller(controller, patient_name, env.sample_time, config)
    trace = PatientTrace(
        name=patient_name if trial is None else f"trial{trial:03d}_{patient_name}",
        trial=trial,
        patient=patient_name,
        meals=protocol.meals,
    )

    n_steps = int(protocol.hours * 60 / env.sample_time)
    for _ in range(n_steps):
        cgm = float(step.observation.CGM)
        action = basal_controller.policy(cgm)
        step = env.step(action)
        trace.time_min.append((len(trace.time_min) + 1) * float(env.sample_time))
        trace.cgm.append(cgm)
        trace.basal.append(float(action.basal))
        trace.bolus.append(float(action.bolus))
        trace.bg.append(float(step.info["bg"]))
    return trace


def run_cohort(
    patients: Iterable[str] = COHORT,
    *,
    protocol: Protocol = Protocol(),
    config: BasalPolicyConfig | Mapping[str, BasalPolicyConfig] = BasalPolicyConfig(),
) -> CohortTrace:
    return CohortTrace(
        tuple(
            run_patient(name, protocol=protocol, config=_config_for_patient(config, name))
            for name in patients
        )
    )


def run_trials(
    preset: str,
    *,
    n_trials: int = 100,
    seed: int = 20260706,
    patients: Iterable[str] = COHORT,
) -> CohortTrace:
    patient_list = list(patients)
    config = preset_config(preset)
    controller = preset_controller(preset)
    specs = randomized_trial_specs(n_trials, seed=seed, patients=patient_list)
    return CohortTrace(
        tuple(
            run_patient(
                spec.patient,
                protocol=spec.protocol,
                config=_config_for_patient(config, spec.patient),
                controller=controller,
                trial=spec.trial,
            )
            for spec in specs
        )
    )


def randomized_trial_specs(
    n_trials: int,
    *,
    seed: int = 20260706,
    patients: Iterable[str] = COHORT,
) -> tuple[TrialSpec, ...]:
    patient_list = list(patients)
    rng = np.random.default_rng(seed)
    patient_order = [patient_list[i % len(patient_list)] for i in range(n_trials)]
    rng.shuffle(patient_order)
    return tuple(
        TrialSpec(
            trial=i,
            patient=patient,
            protocol=randomized_protocol(int(rng.integers(0, 2**31 - 1))),
        )
        for i, patient in enumerate(patient_order, start=1)
    )


def randomized_protocol(seed: int) -> Protocol:
    rng = np.random.default_rng(seed)

    breakfast_time = _clipped_normal(rng, mean=1.0, sd=0.45, lo=0.25, hi=2.75)
    lunch_time = _clipped_normal(rng, mean=6.0, sd=0.7, lo=4.5, hi=7.65)
    breakfast_g = _clipped_normal(rng, mean=45.0, sd=10.0, lo=20.0, hi=75.0)
    lunch_g = _clipped_normal(rng, mean=70.0, sd=12.0, lo=35.0, hi=105.0)
    meals = [(breakfast_time, breakfast_g), (lunch_time, lunch_g)]
    if rng.random() < 0.30:
        snack_time = _clipped_normal(rng, mean=3.5, sd=0.35, lo=3.0, hi=4.25)
        snack_g = _clipped_normal(rng, mean=12.0, sd=5.0, lo=4.0, hi=25.0)
        meals.append((snack_time, snack_g))
    meals = sorted((round(t, 3), round(g, 1)) for t, g in meals)
    return Protocol(meals=tuple(meals), seed=seed)


def preset_config(name: str) -> BasalPolicyConfig | dict[str, BasalPolicyConfig]:
    if name == "clock-prior-aggressive":
        return BasalPolicyConfig()
    if name == "clock-prior-safer":
        return _safer_config()
    if name == "personalized-v1":
        aggressive = BasalPolicyConfig()
        safer = _safer_config()
        sensitive = {
            "adult#003",
            "adult#004",
            "adult#006",
            "adult#007",
            "adolescent#003",
        }
        return {patient: safer if patient in sensitive else aggressive for patient in COHORT}
    if name == "personalized-v2":
        aggressive = BasalPolicyConfig()
        configs = {patient: aggressive for patient in COHORT}
        configs.update(
            {
                "adult#003": _safer_config(),
                "adult#004": _scheduled_config(8.0, 10.0, 120.0, 115.0, 12.0, 165.0),
                "adult#006": _safer_config(),
                "adult#007": _scheduled_config(8.0, 10.0, 120.0, 115.0, 12.0, 165.0),
                "adolescent#002": _resistant_config(18.0, 30.0, 30.0),
                "adolescent#003": _scheduled_config(2.0, 12.0, 125.0, 120.0, 12.0, 170.0),
                "adolescent#005": _scheduled_config(10.0, 15.0, 115.0, 110.0, 18.0, 160.0),
                "adolescent#008": _resistant_config(25.0, 40.0, 40.0),
                "adolescent#009": _scheduled_config(4.0, 18.0, 125.0, 120.0, 18.0, 170.0),
            }
        )
        return configs
    if name == "reactive-physiology-v1":
        return BasalPolicyConfig()
    if name == "randomized-hazard-v1":
        return BasalPolicyConfig(
            windows=(
                MealPriorWindow(center_min=90.0, pre_min=75.0, post_min=90.0, multiplier=10.0),
                MealPriorWindow(center_min=360.0, pre_min=90.0, post_min=120.0, multiplier=12.0),
            ),
            suspend_cgm=110.0,
            predicted_low=105.0,
            falling_cgm=135.0,
            down_cgm=155.0,
            meal_multiplier=18.0,
            basal_multipliers=(0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 18.0, 25.0),
            high_gain=0.06,
            trend_gain=1.7,
            iob_gain=0.035,
        )
    if name == "randomized-hazard-safe-v1":
        return BasalPolicyConfig(
            windows=(
                MealPriorWindow(center_min=90.0, pre_min=75.0, post_min=90.0, multiplier=6.0),
                MealPriorWindow(center_min=360.0, pre_min=90.0, post_min=120.0, multiplier=8.0),
            ),
            suspend_cgm=120.0,
            predicted_low=115.0,
            falling_cgm=145.0,
            down_cgm=165.0,
            meal_multiplier=14.0,
            basal_multipliers=(0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0),
            high_gain=0.045,
            trend_gain=1.2,
            iob_gain=0.05,
        )
    raise ValueError(f"unknown preset: {name}")


def preset_controller(name: str) -> str:
    if name == "reactive-physiology-v1":
        return "reactive-physiology"
    return "clock"


def _safer_config() -> BasalPolicyConfig:
    return BasalPolicyConfig(
        windows=(
            MealPriorWindow(center_min=60.0, pre_min=60.0, post_min=90.0, multiplier=8.0),
            MealPriorWindow(center_min=300.0, pre_min=60.0, post_min=90.0, multiplier=12.0),
        ),
        suspend_cgm=115.0,
        predicted_low=110.0,
        falling_cgm=140.0,
        down_cgm=155.0,
        meal_multiplier=15.0,
        basal_multipliers=(0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 18.0),
    )


def _scheduled_config(
    first_multiplier: float,
    second_multiplier: float,
    suspend_cgm: float,
    predicted_low: float,
    max_multiplier: float,
    down_cgm: float,
) -> BasalPolicyConfig:
    base = BasalPolicyConfig()
    return BasalPolicyConfig(
        windows=(
            MealPriorWindow(
                center_min=60.0,
                pre_min=60.0,
                post_min=90.0,
                multiplier=first_multiplier,
            ),
            MealPriorWindow(
                center_min=300.0,
                pre_min=60.0,
                post_min=90.0,
                multiplier=second_multiplier,
            ),
        ),
        suspend_cgm=suspend_cgm,
        predicted_low=predicted_low,
        falling_cgm=suspend_cgm + 25.0,
        down_cgm=down_cgm,
        meal_multiplier=second_multiplier,
        basal_multipliers=tuple(m for m in base.basal_multipliers if m <= max_multiplier),
    )


def _resistant_config(
    first_multiplier: float, second_multiplier: float, max_multiplier: float
) -> BasalPolicyConfig:
    base = BasalPolicyConfig()
    actions = tuple(sorted({*base.basal_multipliers, 40.0, 50.0, 60.0}))
    return BasalPolicyConfig(
        basal_multipliers=tuple(m for m in actions if m <= max_multiplier),
        windows=(
            MealPriorWindow(
                center_min=60.0,
                pre_min=90.0,
                post_min=150.0,
                multiplier=first_multiplier,
            ),
            MealPriorWindow(
                center_min=300.0,
                pre_min=90.0,
                post_min=150.0,
                multiplier=second_multiplier,
            ),
        ),
        meal_multiplier=second_multiplier,
        iob_gain=0.015,
    )


def _config_for_patient(
    config: BasalPolicyConfig | Mapping[str, BasalPolicyConfig], patient_name: str
) -> BasalPolicyConfig:
    if isinstance(config, BasalPolicyConfig):
        return config
    return config[patient_name]


def _build_controller(
    controller: str,
    patient_name: str,
    sample_min: float,
    config: BasalPolicyConfig,
):
    if controller == "clock":
        return ClockPriorBasalController(patient_name, sample_min, config)
    if controller == "reactive-physiology":
        return ReactivePhysiologyBasalController(patient_name, sample_min, config)
    raise ValueError(f"unknown controller: {controller}")


def write_artifacts(
    result: CohortTrace,
    out_dir: Path,
    *,
    preset: str,
    protocol: Protocol,
    note: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    (out_dir / "summary.txt").write_text(result.summary() + "\n")
    (out_dir / "note.txt").write_text(note.rstrip() + "\n")
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "preset": preset,
                "protocol": {
                    "start": protocol.start.isoformat(),
                    "hours": protocol.hours,
                    "meals": protocol.meals,
                    "seed": protocol.seed,
                },
                "n_traces": len(result.patients),
                "metrics": {
                    "tir": result.tir,
                    "time_low": result.time_low,
                    "time_high": result.time_high,
                    "critical_time": result.critical_time,
                    "critical_samples": result.critical_samples,
                },
            },
            indent=2,
        )
        + "\n"
    )

    with (out_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "name",
                "trial",
                "patient",
                "meals",
                "tir",
                "time_low",
                "time_high",
                "critical_time",
                "critical_samples",
                "min_bg",
                "max_bg",
            ]
        )
        for patient in result.patients:
            writer.writerow(
                [
                    patient.name,
                    patient.trial,
                    patient.patient or patient.name,
                    json.dumps(patient.meals),
                    patient.tir,
                    patient.time_low,
                    patient.time_high,
                    patient.critical_time,
                    patient.critical_samples,
                    patient.min_bg,
                    patient.max_bg,
                ]
            )

    with (out_dir / "traces.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "name",
                "trial",
                "patient",
                "time_min",
                "bg",
                "cgm",
                "basal",
                "bolus",
                "cumulative_tir",
            ]
        )
        for patient in result.patients:
            in_range = 0
            for i, (time_min, bg, cgm, basal, bolus) in enumerate(
                zip(patient.time_min, patient.bg, patient.cgm, patient.basal, patient.bolus),
                start=1,
            ):
                in_range += RANGE_LO <= bg <= RANGE_HI
                writer.writerow(
                    [
                        patient.name,
                        patient.trial,
                        patient.patient or patient.name,
                        time_min,
                        bg,
                        cgm,
                        basal,
                        bolus,
                        in_range / i,
                    ]
                )

    for patient in result.patients:
        _plot_patient(patient, plots_dir / f"{patient.name}.png", protocol)


def _plot_patient(patient: PatientTrace, path: Path, protocol: Protocol) -> None:
    import matplotlib.pyplot as plt

    times = np.asarray(patient.time_min)
    bg = np.asarray(patient.bg)
    cgm = np.asarray(patient.cgm)
    basal = np.asarray(patient.basal)
    cumulative_tir = np.cumsum((RANGE_LO <= bg) & (bg <= RANGE_HI)) / np.arange(1, len(bg) + 1)

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(patient.name)

    axes[0].plot(times, bg, label="BG", color="#1f77b4")
    axes[0].plot(times, cgm, label="CGM", color="#ff7f0e", alpha=0.75)
    axes[0].axhspan(RANGE_LO, RANGE_HI, color="#2ca02c", alpha=0.12)
    axes[0].axhline(CRITICAL_LO, color="#d62728", linewidth=1, linestyle=":")
    axes[0].axhline(CRITICAL_HI, color="#d62728", linewidth=1, linestyle=":")
    axes[0].set_ylabel("mg/dL")
    axes[0].legend(loc="upper left")

    axes[1].plot(times, cumulative_tir, color="#2ca02c")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_ylabel("cum TIR")

    axes[2].step(times, basal, where="post", color="#9467bd", label="basal U/min")
    axes[2].set_ylabel("basal")
    axes[2].set_xlabel("episode minute")

    for axis in axes:
        for meal_hour, grams in patient.meals or protocol.meals:
            axis.axvline(meal_hour * 60, color="#111111", linewidth=1, alpha=0.35)
            axis.text(meal_hour * 60, axis.get_ylim()[1], f"{grams:g}g", va="top", ha="right", fontsize=8)
        axis.grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _fraction(values: list[float], predicate) -> float:
    return sum(1 for value in values if predicate(value)) / len(values)


def _nearest_action(value: float, actions: tuple[float, ...]) -> float:
    return min(actions, key=lambda action: abs(action - value))


def _clipped_normal(
    rng: np.random.Generator, *, mean: float, sd: float, lo: float, hi: float
) -> float:
    return float(np.clip(rng.normal(mean, sd), lo, hi))


def _build_env(patient_name: str, protocol: Protocol) -> T1DSimEnv:
    patient = T1DPatient.withName(patient_name)
    sensor = CGMSensor.withName("Dexcom", seed=protocol.seed)
    pump = InsulinPump.withName("Insulet")
    scenario = CustomScenario(start_time=protocol.start, scenario=list(protocol.meals))
    return T1DSimEnv(patient, sensor, pump, scenario)


def _patient_basal(patient_name: str) -> float:
    params_path = resources.files("simglucose") / "params" / "vpatient_params.csv"
    params = pd.read_csv(str(params_path))
    patient = params[params.Name == patient_name]
    if len(patient) != 1:
        raise ValueError(f"unknown simglucose patient: {patient_name}")
    return float(patient.u2ss.iloc[0]) * float(patient.BW.iloc[0]) / 6000.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        default="clock-prior-aggressive",
        choices=[
            "clock-prior-aggressive",
            "clock-prior-safer",
            "personalized-v1",
            "personalized-v2",
            "reactive-physiology-v1",
            "randomized-hazard-v1",
            "randomized-hazard-safe-v1",
        ],
    )
    parser.add_argument("--trials", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument(
        "--patients",
        nargs="*",
        default=COHORT,
        help="Patient names to run. Defaults to all adults and adolescents.",
    )
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--note", default="")
    args = parser.parse_args(argv)
    protocol = Protocol()
    if args.trials:
        result = run_trials(
            args.preset,
            n_trials=args.trials,
            seed=args.seed,
            patients=args.patients,
        )
    else:
        result = run_cohort(
            args.patients,
            protocol=protocol,
            config=preset_config(args.preset),
        )
    print(result.summary())
    if args.out_dir is not None:
        note = args.note or (
            f"Preset: {args.preset}\n"
            "Approach: basal-only CGM controller with hidden meal clock prior, "
            "trend feedback, predicted-low suspend, and discrete temp-basal actions.\n"
            "Learnings: inspect summary.txt and plots/ for per-patient failure modes.\n"
        )
        write_artifacts(result, args.out_dir, preset=args.preset, protocol=protocol, note=note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
