"""Frozen design, process environment and artifact bookkeeping for the extraction study."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_ROOT = Path.home() / "agent-share/pcgym-extraction-rxinfer"
EXPERIMENT = Path(os.environ.get("PCGYM_EXPERIMENT", "~/experiments/cffg-testing/pc-gym/00/cffg")).expanduser()
JULIA_PROJECT = Path(os.environ.get("PCGYM_RXINFER_PROJECT", "~/agent-share/pcgym-four-library-timing/rxinfer")).expanduser()
DEVELOPMENT = (17101, 17102, 17103)
ENDURANCE = 17104
HELDOUT = tuple(range(17301, 17321))
CONDITIONS = ("stationary", "changing")
ARMS = ("frozen", "informed", "adaptive")
LIBRARIES = ("cffg", "rxinfer")
MODES = ("tape", "control")
HORIZON = 6
DEADLINE = .5
MOTION_SD = .006
PRACTICAL = .0002


def configure():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                 "MKL_NUM_THREADS", "JULIA_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ.update(JAX_ENABLE_X64="1", JAX_PLATFORMS="cpu",
                      XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
    site = EXPERIMENT / f".venv/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    if not site.is_dir():
        raise FileNotFoundError(site)
    sys.path.append(str(site))


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def save(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(record, sort_keys=True, allow_nan=False, indent=2) + "\n")
    os.replace(tmp, path)


def read(path):
    return Path(path).is_file() and json.loads(Path(path).read_text())


def sources():
    files = [*HERE.glob("*.py"), *HERE.glob("*.jl"), *HERE.glob("*.c"), *HERE.glob("*.cpp"),
             *REPO.joinpath("src/cffg").rglob("*.py"), REPO / "uv.lock"]
    return {str(f.relative_to(REPO)): sha(f) for f in sorted(files)
            if f.is_file() and f.name not in ("report.py", "report_check.py")}


def protocol():
    return dict(plant="PCGym multistage_extraction (native balances integrated by CasADi)",
                physical_states=10, observed=["X5", "Y1"], output_order=[8, 1], learned="log(Kla)",
                development_seeds=DEVELOPMENT, endurance_seed=ENDURANCE, reserved_heldout_seeds=HELDOUT,
                short_steps=600, long_steps=6000, dt_model_hours=.05, horizon=HORIZON,
                sensor_sd=[.002,.003], process_sd=.001, prior_x_sd=.035, prior_log_k_sd=.1,
                adaptive_log_motion_sd=MOTION_SD, target_sd=.012, nominal_flow=[75,30],
                action_prior_sd=[15,8], flow_low=[45,18], flow_high=[105,45], deadline_s=DEADLINE,
                modes=MODES, arms=ARMS, conditions=CONDITIONS, libraries=LIBRARIES,
                primary="Mean paired adaptive tape RMSE difference over eight hidden concentrations on changed transitions (CFFG minus RxInfer); lower error wins.",
                practical_margin_concentration=PRACTICAL, inference="Within ±0.0002 for the entire whole-seed interval is a practical tie. Entirely beyond either margin favors the corresponding library. Otherwise inconclusive.",
                gates=dict(affine="Both implementations match dense Gaussian filter and two-action planner.",
                           plant="Live-parameter integration, positive state domain, mask and independent inputs pass.",
                           information="On changed tape, informed hidden RMSE improves over frozen by at least 5% on average and on at least 2/3 development seeds in each library; sensor updates improve on a no-sensor observer.",
                           nonlinear="Finite nonlinear particle reference and stable outlet/hidden predictions on development fixture.",
                           reliability="At least 99% of short and endurance control decisions run without failure/hold in each adaptive library and the finite fallback always scores.",
                           feasible="Complete an uninterrupted 6000-tick run per library, then size the fresh study using observed cost/precision."),
                timing="Warm decision from assimilating latest observation + building local model, solving planner, bounds and two-action readout. Native sim, separate scoring and archive IO excluded. Deadline 0.5 s.",
                failures="On update failure, carry the last finite prediction, else last finite belief, score it and flag the tick. On planning or previous update failure or lateness, apply the previous feasible command. Unavailable predictive densities flagged.",
                selection="Matched native affine Gaussian updates with shared first-order nonlinear dynamics. No additional approximation search.",
                tuning="One design for both libraries. Revisions to a failed development design are recorded and requalified before freeze.",
                interval="Resample whole paired seeds with replacement, 10000 draws; no pooling ticks as independent replicates.")


def inventory(root, stage, seeds):
    return [Path(root)/stage/f"seed-{seed}"/cond/f"{mode}-{arm}-{lib}.npz"
            for seed in seeds for cond in CONDITIONS for mode in MODES for arm in ARMS for lib in LIBRARIES]
