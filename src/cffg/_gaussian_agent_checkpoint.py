"""Typed v2 checkpoints for Gaussian boundaries, including legacy affine v2."""

from dataclasses import asdict
import json

import numpy as np

from cffg.gaussian_agent import GaussianDecision, _Pending, _array, _moments
from cffg.learn import Learnables
from cffg.neural_artifacts import array_digest, write_artifact
from cffg.runtime_types import FailureDiagnostic, RunStatus


_BASE_ARRAYS = {"mean", "cov", "applied", "proposal"}
_BASE_META = {"version", "identity", "tick", "episode_tick", "preview", "energy"}


def _digest(meta, arrays):
    encoded = np.asarray(json.dumps(meta, sort_keys=True, allow_nan=False))
    return array_digest({**arrays, "metadata": encoded})


def save(agent, path):
    pending = agent._pending
    decision = pending.decision if pending else None
    arrays = {
        "mean": agent._moments[0], "cov": agent._moments[1],
        "applied": agent._command_array(agent._applied),
        "proposal": agent._command_array(decision.command if decision else None),
    }
    meta = dict(
        version=2, identity=agent._identity, tick=agent.tick, episode_tick=agent.episode_tick,
        preview=pending.preview if pending else None, energy=decision.free_energy if decision else None,
    )
    if not agent._legacy or (pending and pending.feedback is not None):
        probabilities = None if decision is None else decision.probabilities
        arrays["probabilities"] = np.empty(0, np.float64) if probabilities is None else probabilities
        arrays.update({f"learn:{key}": value for key, value in agent.learnables.state_dict().items()})
        meta.update(
            mode="hybrid" if agent._hybrid else "continuous",
            status=decision.status.value if decision else None,
            held=decision.held if decision else False,
            failure=asdict(decision.failure) if decision and decision.failure else None,
            feedback=pending.feedback if pending else None,
            rng=agent._rng.bit_generator.state,
        )
    meta["digest"] = _digest(meta, arrays)
    write_artifact(path, meta, arrays)


def _hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _empty(value, dtype, label):
    _array(value, (0,), dtype, label, exact=True)
    return None


def _rng(data):
    rng = np.random.default_rng(0)
    if not isinstance(data, dict) or data.keys() != rng.bit_generator.state.keys():
        raise ValueError("invalid selection RNG state")
    state = data["state"]
    if (data["bit_generator"] != "PCG64" or not isinstance(state, dict)
            or set(state) != {"state", "inc"}
            or any(type(value) is not int or not 0 <= value < 2**128 for value in state.values())
            or not state["inc"] % 2
            or type(data["has_uint32"]) is not int or data["has_uint32"] not in (0, 1)
            or type(data["uinteger"]) is not int or not 0 <= data["uinteger"] < 2**32):
        raise ValueError("invalid selection RNG state")
    rng.bit_generator.state = data
    return rng


def _diagnostic(data):
    if data is None:
        return None
    if (not isinstance(data, dict) or set(data) != {"reason", "factor", "phase", "iteration"}
            or not isinstance(data["reason"], str) or not data["reason"]
            or not isinstance(data["phase"], str) or not data["phase"]
            or (data["factor"] is not None and not isinstance(data["factor"], str))
            or type(data["iteration"]) is not int or data["iteration"] < 0):
        raise ValueError("invalid pending decision diagnostic")
    return FailureDiagnostic(**data)


def _stage_pending(agent, meta, arrays, *, legacy):
    preview, energy = meta["preview"], meta["energy"]
    status = RunStatus.CONVERGED.value if legacy and preview is not None else meta.get("status")
    held, failure, feedback = meta.get("held", False), meta.get("failure"), meta.get("feedback")
    probability_array = arrays.get("probabilities", np.empty(0, np.float64))
    dtype = np.dtype(np.int64) if agent._hybrid else agent._dtype
    if preview is None:
        if energy is not None or status is not None or held is not False or failure is not None or feedback is not None:
            raise ValueError("invalid pending Gaussian decision")
        _empty(arrays["proposal"], dtype, "proposal")
        _empty(probability_array, np.float64, "action probabilities")
        return None

    if (not _hash(preview) or (feedback is not None and not _hash(feedback))
            or type(held) is not bool or not isinstance(status, str)):
        raise ValueError("invalid pending Gaussian decision")
    status = RunStatus(status)
    command = agent._command(arrays["proposal"], "proposal", exact=True)
    failure = _diagnostic(failure)
    if held:
        if (agent._on_failure != "hold" or status is RunStatus.CONVERGED
                or energy is not None or failure is None):
            raise ValueError("invalid held decision")
        expected = agent._initial_command if meta["episode_tick"] == 0 else arrays["applied"]
        if not np.array_equal(command, expected):
            raise ValueError("held decision differs from the last applied command")
    elif (status is not RunStatus.CONVERGED or type(energy) is not float
          or not np.isfinite(energy) or failure is not None):
        raise ValueError("invalid successful pending decision")

    if agent._hybrid and not held:
        q = _array(probability_array, (agent._nu,), np.float64, "action probabilities", exact=True)
        if np.any(q < 0) or abs(q.sum() - 1.) > 4 * len(q) * np.finfo(q.dtype).eps or q[command] == 0:
            raise ValueError("invalid pending categorical probabilities")
        if agent._select == "map" and command != int(np.argmax(q)):
            raise ValueError("pending command differs from the categorical mode")
    else:
        q = _empty(probability_array, np.float64, "action probabilities")
    return _Pending(preview, GaussianDecision(meta["tick"], command, energy, status, q, held, failure), feedback)


def restore(agent, path):
    with np.load(path, allow_pickle=False) as record:
        fields = set(record.files)
        if len(fields) != len(record.files) or "metadata" not in fields:
            raise ValueError("invalid Gaussian agent checkpoint inventory")
        raw = record["metadata"]
        if raw.shape != () or raw.dtype.kind != "U":
            raise ValueError("invalid Gaussian agent checkpoint metadata")
        meta = json.loads(str(raw))
        arrays = {name: record[name] for name in fields - {"metadata"}}
    if (not isinstance(meta, dict) or type(meta.get("version")) is not int
            or meta["version"] != 2 or meta.get("identity") != agent._identity):
        raise ValueError("incompatible Gaussian agent checkpoint")
    legacy = "mode" not in meta
    metadata = set(_BASE_META)
    expected_arrays = set(_BASE_ARRAYS)
    if legacy:
        if not agent._legacy:
            raise ValueError("incompatible legacy Gaussian agent checkpoint")
    else:
        if meta["mode"] != ("hybrid" if agent._hybrid else "continuous"):
            raise ValueError("incompatible Gaussian checkpoint mode")
        metadata.update({"mode", "status", "held", "failure", "feedback", "rng"})
        expected_arrays.update({"probabilities", *(f"learn:{k}" for k in agent.learnables.state_dict())})
    if set(meta) != metadata | {"digest"} or set(arrays) != expected_arrays:
        raise ValueError("invalid Gaussian agent checkpoint inventory")
    if meta.pop("digest") != _digest(meta, arrays):
        raise ValueError("incompatible Gaussian agent checkpoint")

    tick, episode = meta["tick"], meta["episode_tick"]
    if type(tick) is not int or type(episode) is not int or not 0 <= episode <= tick:
        raise ValueError("invalid Gaussian agent cadence")
    moments = _moments(arrays["mean"], arrays["cov"], agent._dim, agent._dtype, exact=True)
    applied_array = arrays["applied"]
    dtype = np.dtype(np.int64) if agent._hybrid else agent._dtype
    applied = (agent._command(applied_array, "previous action", exact=True) if episode
               else _empty(applied_array, dtype, "previous action"))
    pending = _stage_pending(agent, meta, arrays, legacy=legacy)
    learner, rng = agent.learnables, agent._rng
    if not legacy:
        learner = Learnables.from_persistent_state(agent.filter, learner.initial_params, learner.params)
        learner.load_state_dict({
            key.removeprefix("learn:"): value for key, value in arrays.items() if key.startswith("learn:")
        }, exact=True)
        rng = _rng(meta["rng"])
    agent._moments, agent._applied, agent._pending = moments, applied, pending
    agent.learnables, agent._rng = learner, rng
    agent.tick, agent.episode_tick = tick, episode
