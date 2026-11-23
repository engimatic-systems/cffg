"""PCGym CSTR with a scheduled, learner-hidden heat-transfer coefficient."""

import numpy as np

if __package__:
    from .pilot import plant_rng, rng_digest
else:
    from pilot import plant_rng, rng_digest


SWITCH_TICKS = (178, 373)


def _model():
    # The benchmark loads the pinned external PCGym after configuring the runtime.
    from pcgym.model_classes import cstr

    class ScheduledCSTR(cstr):
        def __post_init__(self):
            super().__post_init__()
            self.disturbances = ["UA"]

        def __call__(self, state, inputs):
            # CasADi makes UA a symbolic input alongside the applied coolant.
            return cstr(UA=inputs[1], int_method=self.int_method)(state, inputs[:1])

    return ScheduledCSTR()


def _observation(raw):
    obs = np.array(raw[:2], dtype=float, copy=True)
    if obs.shape != (2,) or not np.isfinite(obs).all():
        raise ValueError("Expected two finite normalized CSTR measurements")
    return obs


class _Plant:
    def __init__(self, env, stream, raw):
        self._env = env
        self._stream = stream
        self._done = False
        self.obs = _observation(raw)

    def step(self, normalized_applied_scalar):
        """Advance once. Only the first returned value is a learner observation."""
        if self._done:
            raise RuntimeError("The plant has already terminated")
        action = np.asarray(normalized_applied_scalar, dtype=float)
        if action.shape != () or not -1. <= action <= 1.:
            raise ValueError("Expected one applied normalized coolant command in [-1, 1]")
        with plant_rng(self._stream):
            raw, reward, terminated, truncated, _ = self._env.step(action.reshape(1))
        self._done = bool(terminated or truncated)
        self.obs = _observation(raw)
        truth = np.array(self._env.state[:2], dtype=float, copy=True)
        reward = float(np.asarray(reward).item())
        if not np.isfinite(truth).all() or not np.isfinite(reward):
            raise ArithmeticError("Nonfinite PCGym truth or reward")
        return self.obs, truth, reward, self._done

    def close(self):
        self._env.close()


def make_plant(make_env, base_params, transition_ua, seed):
    """Use transition_ua[t] for x[t] -> x[t+1]; .obs exposes only noisy Ca and T."""
    ua = np.asarray(transition_ua, dtype=float)
    if (ua.shape != (base_params["N"] - 1,) or not ua.size
            or not np.isfinite(ua).all() or not (ua > 0).all()):
        raise ValueError("Expected one finite positive UA per transition")
    if (base_params["model"] != "cstr" or base_params["integration_method"] != "casadi"
            or not base_params["normalise_a"] or not base_params["normalise_o"]
            or base_params.get("a_delta", False)):
        raise ValueError("Expected the CasADi CSTR with normalized observations and absolute coolant commands")
    # PCGym stores the first disturbance at reset and feeds disturbance[t+1] to the integrator.
    configured = {**base_params, "custom_model": _model(), "disturbances": {"UA": np.r_[ua[0], ua]},
                  "disturbance_bounds": {"low": np.array([0.]), "high": np.array([ua.max()])}}
    stream = np.random.RandomState(seed)
    with plant_rng(stream):
        env = make_env(configured)
        try:
            raw, _ = env.reset(seed=seed)
            return _Plant(env, stream, raw)
        except Exception:
            env.close()
            raise


def _fixed_step(make_env, base_params, ua, state, command):
    """One ordinary PCGym step from the identical physical state with fixed UA."""
    from pcgym.model_classes import cstr

    x0 = np.array(base_params["x0"], dtype=float, copy=True)
    x0[:2] = state
    params = {**base_params, "custom_model": cstr(UA=float(ua)), "x0": x0, "noise": False}
    with plant_rng(np.random.RandomState(0)):
        env = make_env(params)
        try:
            env.reset(seed=0)
            env.step(np.array([command]))
            return np.array(env.state[:2], copy=True)
        finally:
            env.close()


def _qualify(make_env, params, ua, switches):
    """Audit an uninterrupted prefix through both switches, using one-step references."""
    changes = np.flatnonzero(ua[1:] != ua[:-1]) + 1
    if changes.size and (tuple(changes) != switches or ua[-1] != ua[0]):
        raise ValueError(f"Expected stationary UA or a change and return at {switches}")
    if not params["noise"] or not 0 < params["noise_percentage"] < 1:
        raise ValueError("Qualification requires multiplicative sensor noise")
    seed = 16201
    checkpoints = {0, *(t + offset for t in switches for offset in (-1, 0, 1))}
    ambient = rng_digest(np.random)
    expected_rng = np.random.RandomState(seed)
    plant = make_plant(make_env, params, ua, seed)
    # The Gym bounds were rounded to float32 before PCGym normalized the sensors.
    low, high = (np.asarray(params["o_space"][k], dtype=np.float32)[:2] for k in ("low", "high"))
    max_noise_error = 0.
    records = []
    env = plant._env
    integrator = env.int_eng
    # Compare one-step temperatures and concentrations in their own physical units.
    rtol, atol = 1e-9, 1e-11

    def check_obs(observation, truth, noise):
        if (observation.shape != (2,) or env.normobs.shape != (4,)
                or not np.array_equal(observation, env.normobs[:2])
                or np.shares_memory(observation, env.normobs)):
            raise ValueError("Scheduled disturbance or setpoint leaked through the observation boundary")
        decoded = low + (observation + 1) * (high - low) / 2
        error = float(np.max(abs(decoded / truth - 1 - noise)))
        if not error <= 32 * np.finfo(float).eps:
            raise ValueError(f"Sensor stream differs at state index {env.t}")
        return error

    try:
        previous = np.array(params["x0"][:2], dtype=float)
        max_noise_error = check_obs(plant.obs, previous, np.zeros(2))
        for t in range(max(checkpoints) + 1):
            if env.t != t or env.int_eng is not integrator or not np.array_equal(env.state[:2], previous):
                raise ValueError(f"Plant clock, integrator, or physical state discontinuity at transition {t}")
            command = float(.35 * np.sin(2 * np.pi * t / 73))
            obs, truth, _, done = plant.step(command)
            if env.t != t + 1 or env.state[-1] != ua[t] or done or env.int_eng is not integrator:
                raise ValueError(f"Incorrect disturbance, clock or termination at transition {t}")
            noise = expected_rng.normal(size=2) * params["noise_percentage"]
            max_noise_error = max(max_noise_error, check_obs(obs, truth, noise))
            if t in checkpoints:
                fixed = _fixed_step(make_env, params, ua[t], previous, command)
                if not np.allclose(truth, fixed, rtol=rtol, atol=atol):
                    raise ValueError(f"Scheduled and fixed-UA dynamics differ at transition {t}")
                row = dict(t=t, ua=float(ua[t]), applied=command, state_before=previous.tolist(),
                           state_after=truth.tolist(), fixed_ua_after=fixed.tolist(),
                           abs_error_Ca_T=np.abs(truth - fixed).tolist())
                if t in changes:
                    wrong = _fixed_step(make_env, params, ua[t - 1], previous, command)
                    if np.allclose(truth, wrong, rtol=rtol, atol=atol):
                        raise ValueError(f"UA switch had no distinguishable physical effect at transition {t}")
                    row.update(previous_ua=float(ua[t - 1]), previous_ua_after=wrong.tolist())
                records.append(row)
            previous = truth
        sensor_hash = rng_digest(plant._stream)
        if sensor_hash != rng_digest(expected_rng):
            raise ValueError("Final PCGym RNG state differs from serial sensor draws")
    finally:
        plant.close()
    if rng_digest(np.random) != ambient:
        raise ValueError("Plant qualification altered the caller's NumPy RNG")
    return dict(passed=True, switch_ticks=changes.tolist(), probe_ticks=sorted(checkpoints),
                transitions_checked=max(checkpoints) + 1, trajectory_resets=1,
                continuous_state_time_and_integrator=True, learner_observation=["Ca_norm", "T_norm"],
                raw_observation_width=4, ambient_rng_preserved=True, noise_seed=seed,
                noise_rng_sha256=sensor_hash, max_sensor_noise_replay_error=max_noise_error,
                fixed_comparison_rtol=rtol, fixed_comparison_atol=atol, fixed_ua_probes=records)


def qualification(make_env, base_params, transition_ua):
    """Return a JSON-ready audit or raise if the 599-transition pilot plant fails."""
    if base_params["N"] != 600:
        raise ValueError("Expected the existing 599-transition PCGym parameters")
    ua = np.asarray(transition_ua, dtype=float)
    if ua.shape != (599,):
        raise ValueError("Expected 599 transition-indexed UA values")
    return _qualify(make_env, base_params, ua, SWITCH_TICKS)
