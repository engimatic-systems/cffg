"""The simglucose time-in-range benchmark (side project).

Benchmark definition (from the overnight report in ~/agent-share, kept
compatible so numbers are comparable):

- **Unannounced meals**: the controller sees CGM only — no meal signal. The
  announced basal-bolus controller is a ~96% TIR ceiling reference; unannounced
  basal-only is the ~34% floor.
- Fixed scenario: 45 g at 07:00, 70 g at 12:00, 80 g at 18:00; Dexcom sensor;
  Insulet pump; 24 h; adult cohort.
- Metric: fraction of simulator *BG* (not CGM) in [70, 180] mg/dL. FDA-cited
  target: TIR ≥ 70%.

The `reactive` controller below is the current in-repo contender: patient
basal + CGM-trend meal detection + proportional corrections, quest-scaled.
Optimizing this benchmark (CFFG agent) is a later stage.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from datetime import datetime, timedelta


def _shim_legacy_gym() -> None:
    """simglucose's package __init__ registers a legacy-gym env; gym 0.9.4
    cannot even import on Python 3.12 (distutils, pkg_resources). Only the
    registration call and the gym wrapper touch gym — the simulation core is
    gym-free — so a no-op stub is faithful."""
    try:
        import gym  # noqa: F401
    except Exception:
        gym = types.ModuleType("gym")
        envs = types.ModuleType("gym.envs")
        reg = types.ModuleType("gym.envs.registration")
        reg.register = lambda **kwargs: None
        envs.registration = reg
        gym.envs = envs
        sys.modules.update({"gym": gym, "gym.envs": envs, "gym.envs.registration": reg})


_shim_legacy_gym()

from simglucose.actuator.pump import InsulinPump  # noqa: E402
from simglucose.controller.base import Action  # noqa: E402
from simglucose.controller.basal_bolus_ctrller import BBController  # noqa: E402
from simglucose.patient.t1dpatient import T1DPatient  # noqa: E402
from simglucose.sensor.cgm import CGMSensor  # noqa: E402
from simglucose.simulation.env import T1DSimEnv  # noqa: E402
from simglucose.simulation.scenario import CustomScenario  # noqa: E402

START = datetime(2018, 1, 1, 0, 0)
MEALS = [(7, 45), (12, 70), (18, 80)]  # (hour, grams)
ADULTS = [f"adult#{i:03d}" for i in range(1, 11)]
RANGE_LO, RANGE_HI = 70.0, 180.0


@dataclass
class PatientResult:
    name: str
    bg: list[float] = field(default_factory=list)

    @property
    def tir(self) -> float:
        inr = sum(1 for g in self.bg if RANGE_LO <= g <= RANGE_HI)
        return inr / len(self.bg)

    @property
    def time_low(self) -> float:
        return sum(1 for g in self.bg if g < RANGE_LO) / len(self.bg)

    @property
    def time_high(self) -> float:
        return sum(1 for g in self.bg if g > RANGE_HI) / len(self.bg)


@dataclass
class CohortResult:
    patients: list[PatientResult]

    @property
    def tir(self) -> float:
        return sum(p.tir for p in self.patients) / len(self.patients)

    @property
    def time_low(self) -> float:
        return sum(p.time_low for p in self.patients) / len(self.patients)

    @property
    def time_high(self) -> float:
        return sum(p.time_high for p in self.patients) / len(self.patients)

    def summary(self) -> str:
        lines = [
            f"  {p.name}: TIR {p.tir:6.1%}  low {p.time_low:5.1%}  high {p.time_high:5.1%}"
            for p in self.patients
        ]
        head = f"cohort TIR {self.tir:.2%} (low {self.time_low:.2%}, high {self.time_high:.2%})"
        return "\n".join([head, *lines])


class ReactiveController:
    """CGM-only controller: patient basal + trend-triggered meal boluses +
    gentle proportional corrections, with insulin-on-board lockouts.

    Quest-scaled per patient (CR = carb ratio g/U, CF = correction factor
    mg/dL per U). Deliberately conservative about hypoglycemia: corrections
    stop below 150, boluses scale down against recent insulin."""

    def __init__(self, name: str, sample_min: float):
        import pandas as pd
        from importlib import resources

        quest_path = resources.files("simglucose") / "params" / "Quest.csv"
        quest = pd.read_csv(str(quest_path))
        params_path = resources.files("simglucose") / "params" / "vpatient_params.csv"
        vparams = pd.read_csv(str(params_path))
        q = quest[quest.Name == name]
        p = vparams[vparams.Name == name]
        self.cr = float(q.CR.iloc[0]) if len(q) else 15.0
        self.cf = float(q.CF.iloc[0]) if len(q) else 40.0
        self.basal = float(p.u2ss.iloc[0]) * float(p.BW.iloc[0]) / 6000.0  # U/min
        self.sample_min = sample_min
        self.prev_cgm: float | None = None
        self.iob = 0.0  # crude insulin-on-board, decays per step
        self.iob_decay = 0.5 ** (sample_min / 40.0)  # ~40 min half-life
        self.meal_lockout = 0.0  # minutes remaining

    def policy(self, cgm: float) -> Action:
        trend = 0.0 if self.prev_cgm is None else (cgm - self.prev_cgm) / self.sample_min
        self.prev_cgm = cgm
        self.iob *= self.iob_decay
        self.meal_lockout = max(0.0, self.meal_lockout - self.sample_min)

        bolus_u = 0.0
        # Meal detection: sustained fast rise while not locked out.
        if trend > 1.8 and cgm > 130 and self.meal_lockout == 0.0:
            est_carbs = 55.0
            bolus_u = est_carbs / self.cr
            self.meal_lockout = 90.0
        # Correction: high and not falling, damped by IOB.
        elif cgm > 180 and trend > -0.5:
            want = (cgm - 140.0) / self.cf
            bolus_u = max(0.0, want - self.iob) * 0.35

        # Hypo protection: suspend basal when low or dropping fast toward low.
        basal = self.basal
        if cgm < 90 or (cgm < 110 and trend < -1.0):
            basal = 0.0
            bolus_u = 0.0

        self.iob += bolus_u
        # Pump expects bolus as U/min over one sample interval.
        return Action(basal=basal, bolus=bolus_u / self.sample_min)


def _build_env(name: str, seed: int) -> T1DSimEnv:
    patient = T1DPatient.withName(name)
    sensor = CGMSensor.withName("Dexcom", seed=seed)
    pump = InsulinPump.withName("Insulet")
    scenario = CustomScenario(start_time=START, scenario=MEALS)
    return T1DSimEnv(patient, sensor, pump, scenario)


def run_patient(name: str, controller: str, hours: float = 24.0, seed: int = 7) -> PatientResult:
    env = _build_env(name, seed)
    step = env.reset()
    sample_min = env.sample_time
    result = PatientResult(name=name)

    bb = BBController() if controller in ("announced_bb",) else None
    reactive = ReactiveController(name, sample_min) if controller == "reactive" else None
    basal_only = None
    if controller == "basal_only":
        basal_only = ReactiveController(name, sample_min).basal

    n_steps = int(hours * 60 / sample_min)
    for _ in range(n_steps):
        cgm = step.observation.CGM
        if bb is not None:  # privileged: sees the meal signal
            action = bb.policy(step.observation, 0, False, patient_name=name, meal=step.info.get("meal", 0) if isinstance(step.info, dict) else 0, sample_time=sample_min)
        elif reactive is not None:
            action = reactive.policy(cgm)
        else:
            action = Action(basal=basal_only, bolus=0.0)
        step = env.step(action)
        result.bg.append(float(env.patient.observation.Gsub))
    return result


def run_cohort(
    controller: str, patients: list[str] | None = None, hours: float = 24.0, seed: int = 7
) -> CohortResult:
    return CohortResult([run_patient(p, controller, hours, seed) for p in patients or ADULTS])
