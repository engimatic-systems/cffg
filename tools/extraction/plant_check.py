"""Development-only extraction plant qualification; no competitive library timings."""

import argparse
from itertools import product
from pathlib import Path
from time import perf_counter

if __package__:
    from . import study
else:
    import study

study.configure()

import casadi as ca
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

if __package__:
    from . import plant as p
else:
    import plant as p


def fixed_reference(x, u, kla, dt=p.DT):
    """Independently instantiate ordinary fixed-Kla PCGym and use SciPy's DOP853."""
    from pcgym.model_classes import multistage_extraction

    model = multistage_extraction(Kla=float(kla), int_method="casadi")
    result = solve_ivp(lambda t, v: model(v, u), (0, dt), x, method="DOP853",
                       atol=1e-13, rtol=1e-11)
    if not result.success:
        raise ArithmeticError(result.message)
    return result.y[:, -1]


def diagnostic_transition():
    """Independent differentiable RK4 of the native RHS, used by reference observers."""
    x, u = ca.SX.sym("x", 11), ca.SX.sym("u", 2)
    rhs = ca.Function("rhs", [x, u], [ca.vertcat(*p._native(ca.exp(x[10]))(x[:10], u), 0)])
    y = x
    h = p.DT / 8
    for _ in range(8):
        a = rhs(y, u)
        b = rhs(y + h * a / 2, u)
        c = rhs(y + h * b / 2, u)
        d = rhs(y + h * c, u)
        y = y + h / 6 * (a + 2*b + 2*c + d)
    return ca.Function("discrete", [x, u], [y, ca.jacobian(y, x), ca.jacobian(y, u)])


def central_jacobian(f, v):
    eps = 1e-5 * np.maximum(1., np.abs(v))
    return np.column_stack([(f(v + dv) - f(v - dv)) / (2 * step)
                            for step, dv in zip(eps, np.diag(eps))])


def physical_checks(plant, transition):
    steady = p.nominal_profile()
    extrema = list(product(*zip(p.FLOW_LOW, p.FLOW_HIGH)))
    transition_errors, derivative_errors, truth_errors, reach = [], [], [], {}
    for u in (p.NOMINAL_FLOW, *map(np.array, extrema)):
        for kla in (p.NOMINAL_KLA, p.CHANGED_KLA):
            model = p._native(kla)
            sol = root(lambda x: model(x, u), steady, tol=1e-11)
            assert sol.success and max(abs(sol.fun)) < 1e-9 and np.all((0 < sol.x) & (sol.x < 1))
            reach[f"L={u[0]:g} G={u[1]:g} @ {kla}"] = float(sol.x[8])
            x = steady + np.linspace(-.025, .035, 10)
            z = np.r_[x, np.log(kla)]
            y, a, b = map(np.asarray, transition(z, u))
            v = np.r_[z, u]
            def ref(v):
                return fixed_reference(v[:10], v[11:], np.exp(v[10]))
            yref = ref(v)
            jac = central_jacobian(ref, v)
            transition_errors.append(max(abs(y.ravel()[:10] - yref)))
            truth_errors.append(max(abs(plant.step(x, u, kla) - yref)))
            derivative_errors.append(np.max(abs(np.c_[a, b][:10] - jac)))
    gain = np.asarray(transition(np.r_[steady, np.log(p.NOMINAL_KLA)], p.NOMINAL_FLOW)[1])
    poles = abs(np.linalg.eigvals(gain[:10, :10]))
    settle = float(np.log(.05) / np.log(poles.max()))
    h = np.eye(11)[p.OUTLETS] / p.SENSOR_SD[:, None]
    scales = np.diag([*[p.INITIAL_SD]*10, p.LOG_KLA_SD])
    obs = np.vstack([h @ np.linalg.matrix_power(gain, t) @ scales for t in range(11)])
    assert max(truth_errors) < 1e-8 and max(transition_errors) < 3e-6 and max(derivative_errors) < 5e-5
    # Flow corners applied from nominal for the same finite preview as the planner.
    horizon_x5 = {}
    for kla in (p.NOMINAL_KLA, p.CHANGED_KLA):
        end = [fixed_reference(steady, np.asarray(u), kla, study.HORIZON*p.DT)[8] for u in extrema]
        horizon_x5[str(kla)] = [min(end), max(end)]
        assert min(end) < .295 and max(end) > .375
        limits = [v for key,v in reach.items() if key.endswith(f"@ {kla}")]
        assert min(limits) < .295 and max(limits) > .375
    return dict(truth_dop853_max_abs=max(truth_errors), reference_rk4_state_max_abs=max(transition_errors),
                reference_rk4_jacobian_max_abs=max(derivative_errors), equilibrium_x5=reach,
                nominal_largest_pole=float(poles.max()), nominal_slowest_95pct_ticks=settle,
                nominal_slowest_95pct_hours=settle*p.DT, horizon_hours=study.HORIZON*p.DT, horizon_x5_from_nominal=horizon_x5,
                scaled_observability_singular_values=np.linalg.svd(obs, compute_uv=False).tolist())


def rollout(plant, tape):
    path = [tape["initial"]]
    for u, kla, w in zip(tape["flows"], tape["kla"], tape["process"], strict=True):
        path.append(plant.step(path[-1], u, kla) + w)
    x = np.array(path)
    assert np.all((0 < x) & (x < 1)), "Sampled truth left the declared concentration domain"
    return x


def assimilate(x, cov, reading):
    h = np.eye(len(x))[p.OUTLETS]
    r = np.diag(p.SENSOR_SD**2)
    k = np.linalg.solve(h @ cov @ h.T + r, h @ cov).T
    left = np.eye(len(x)) - k @ h
    return x + k @ (reading - h @ x), left @ cov @ left.T + k @ r @ k.T


def observer(transition, tape, truth, method):
    """Reference EKF; informed gets current Kla, adaptive uses only readings/applied flows."""
    adaptive = method == "adaptive"
    mean = np.r_[p.nominal_profile(), np.log(p.NOMINAL_KLA)] if adaptive else p.nominal_profile()
    cov = np.eye(len(mean)) * p.INITIAL_SD**2
    q = np.eye(len(mean)) * p.PROCESS_SD**2
    if adaptive:
        cov[-1,-1], q[-1,-1] = p.LOG_KLA_SD**2, study.MOTION_SD**2
    if method != "open_loop":
        mean, cov = assimilate(mean, cov, tape["y0"])
    posterior_errors, predictive_errors, sensor_errors, kla_errors = [], [], [], []
    for t, u in enumerate(tape["flows"]):
        kla = tape["kla"][t]
        if adaptive:
            z = mean
        else:
            current = p.NOMINAL_KLA if method == "frozen" else kla
            z = np.r_[mean, np.log(current)]
        kla_errors.append(float(np.exp(z[-1]) - kla))
        y, a, _ = transition(z, u)
        mean = np.asarray(y).ravel()[:len(mean)]
        a = np.asarray(a)[:len(mean), :len(mean)]
        cov = a @ cov @ a.T + q
        reading = p.observe(truth[t+1], tape["noise"][t])
        predictive_errors.append(mean[p.OUTLETS] - truth[t+1, p.OUTLETS])
        sensor_errors.append(mean[p.OUTLETS] - reading)
        if method != "open_loop":
            mean, cov = assimilate(mean, cov, reading)
        posterior_errors.append(mean[:10] - truth[t+1])
    err = np.array(posterior_errors)
    groups = {"all": np.ones(len(err), dtype=bool), "initial_10": np.arange(len(err)) < 10}
    if tape["phase"].any():
        groups["changed"] = tape["phase"] > 0
    scores = {}
    for name, mask in groups.items():
        hidden = err[mask][:,p.HIDDEN]
        outlet = np.array(predictive_errors)[mask]
        scores[name] = dict(hidden_rmse=np.sqrt(np.mean(hidden**2)),
                            state_rmse=np.sqrt(np.mean(err[mask]**2, axis=0)).tolist(),
                            noiseless_outlet_rmse=np.sqrt(np.mean(outlet**2, axis=0)).tolist(),
                            whitened_outlet_rmse=np.sqrt(np.mean((outlet / p.SENSOR_SD)**2)),
                            reading_rmse=np.sqrt(np.mean(np.array(sensor_errors)[mask]**2, axis=0)).tolist(),
                            pre_observation_kla_mae=float(np.mean(abs(np.array(kla_errors)[mask]))))
    return scores


def check_seed(plant, transition, seed, n, observers):
    ch = p.design(seed, n, "changing")
    st = p.design(seed, n, "stationary")
    for key in ("initial", "reset_noise", "y0", "noise", "process", "flows", "target"):
        np.testing.assert_array_equal(ch[key], st[key])
    prefix = p.design(seed, 199, "changing")
    for key in ch:
        np.testing.assert_array_equal(prefix[key], ch[key] if key in ("initial", "y0", "reset_noise") else ch[key][:199])
    np.testing.assert_array_equal(ch["phase"], ch["kla"] < p.NOMINAL_KLA)
    assert set(ch["event"]) == {"nominal","step1","return1","step2","return2",
                                "ramp_down","ramp_hold","ramp_up","return_ramp"}
    for k in (ch, st):
        assert np.all((p.FLOW_LOW <= k["flows"]) & (k["flows"] <= p.FLOW_HIGH))
        assert np.all((p.CHANGED_KLA <= k["kla"]) & (k["kla"] <= p.NOMINAL_KLA))
    assert np.all(st["kla"] == p.NOMINAL_KLA)
    assert np.all(st["event"] == "nominal")
    start = perf_counter()
    truth = rollout(plant, ch)
    wall = perf_counter() - start
    stationary = rollout(plant, st)
    first = np.flatnonzero(ch["phase"])[0]
    np.testing.assert_array_equal(truth[:first+1], stationary[:first+1])
    assert max(abs(truth[first+1] - stationary[first+1])) > 1e-5
    # Replay native fixed-Kla transitions at switch boundaries, through the terminal row.
    change = np.flatnonzero(ch["event"][1:] != ch["event"][:-1]) + 1
    ix = np.unique(np.clip(np.r_[0, n-1, change-1, change, change+1], 0, n-1))
    maxerr = max(np.max(abs(fixed_reference(truth[t], ch["flows"][t], ch["kla"][t])
                            + ch["process"][t] - truth[t+1])) for t in ix)
    assert maxerr < 1e-8
    np.testing.assert_array_equal(ch["y0"], p.observe(truth[0], ch["reset_noise"]))
    yy = np.array([p.observe(v, e) for v, e in zip(truth[1:], ch["noise"], strict=True)])
    np.testing.assert_array_equal(yy, truth[1:, p.OUTLETS] + ch["noise"])
    # Carry an independently integrated state all the way to the terminal sample.
    # Reapply every realized command and shock, reconstruct measurements and
    # post-command tracking costs with the public reference for that transition.
    independent = ch["initial"].copy()
    replay_error = outlet_error = cost_error = 0.
    for t, (flow, kla) in enumerate(zip(ch["flows"], ch["kla"], strict=True)):
        independent = fixed_reference(independent, flow, kla) + ch["process"][t]
        replay_error = max(replay_error, np.max(abs(independent - truth[t+1])))
        measured = independent[[8, 1]] + ch["noise"][t]
        outlet_error = max(outlet_error, np.max(abs(measured - yy[t])))
        target = ch["target"][t]
        cost_error = max(cost_error, abs((independent[8] - target)**2 - (truth[t+1, 8] - target)**2))
    assert max(replay_error, outlet_error, cost_error) < 1e-8
    # The scorer-only interior cannot affect the reset or subsequent sensor output.
    alternate = truth[-1].copy()
    alternate[p.HIDDEN] *= 0.75
    np.testing.assert_array_equal(p.observe(alternate, ch["noise"][-1]), yy[-1])
    # Known-action open-loop outlet separation under matched process innovations.
    changed = ch["phase"] > 0
    separation = (truth[1:,p.OUTLETS] - stationary[1:,p.OUTLETS])[changed] / p.SENSOR_SD
    flows = ch["flows"]
    corr = float(np.corrcoef(flows.T)[0,1])
    assert abs(corr) < .5 and np.all(np.ptp(flows, axis=0) > .75 * (p.FLOW_HIGH-p.FLOW_LOW))
    record = dict(n=n, elapsed_model_hours=n*p.DT, last_state=truth[-1].tolist(),
                  changed_count=int(sum(changed)), max_fixed_kla_step_error=maxerr,
                  replay_max_state_error=replay_error, replay_max_outlet_error=outlet_error,
                  replay_max_squared_tracking_error_difference=cost_error,
                  replay_terminal_error=np.max(abs(independent - truth[-1])),
                  checked_fixed_kla_rows=len(ix), changing_state_range=[truth.min(),truth.max()],
                  stationary_state_range=[stationary.min(),stationary.max()],
                  outlet_separation_sensor_sd_rms=np.sqrt(np.mean(separation**2,axis=0)).tolist(),
                  flow_correlation=corr, unique_flow_counts=[len(np.unique(u)) for u in flows.T],
                  target_range=[ch["target"].min(),ch["target"].max()], native_rollout_seconds=wall)
    if observers:
        record["reference_observers"] = {method: observer(transition, ch, truth, method)
                                         for method in ("informed", "adaptive", "frozen", "open_loop")}
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=study.DEFAULT_ROOT / "qual" / "plant-qualification.json")
    args = parser.parse_args()
    started = perf_counter()
    plant, transition = p.Plant(), diagnostic_transition()
    checks = physical_checks(plant, transition)
    runs = {str(seed): check_seed(plant, transition, seed, 600, True) for seed in study.DEVELOPMENT}
    long = check_seed(plant, transition, study.ENDURANCE, 6000, False)
    gates = {}
    for seed, run in runs.items():
        refs = run["reference_observers"]
        def gain(method, metric, baseline="frozen", period="changed"):
            return 1 - refs[method][period][metric] / refs[baseline][period][metric]
        effects = dict(informed_hidden=gain("informed","hidden_rmse"),
                       informed_outlet=gain("informed","whitened_outlet_rmse"),
                       adaptive_hidden=gain("adaptive","hidden_rmse"),
                       informed_filtered=gain("informed","hidden_rmse","open_loop","all"))
        run["relative_rmse_reductions"] = effects
        gates[seed] = dict(informed_hidden=bool(effects["informed_hidden"] >= .05),
                           informed_outlet=bool(effects["informed_outlet"] >= .05),
                           adaptive_hidden=bool(effects["adaptive_hidden"] > 0),
                           informed_filtered=bool(effects["informed_filtered"] > 0))
    # Equal weight per seed. Small reductions expose internal concentrations for
    # which conditioning on these two sensors adds little beyond known dynamics.
    state_names = [f"{phase}{i}" for i in range(1, 6) for phase in ("X","Y")]
    sensor_gains = {}
    for period in ("all", "changed", "initial_10"):
        def mean_state_rmse(method):
            return np.mean([r["reference_observers"][method][period]["state_rmse"]
                            for r in runs.values()], axis=0)
        reduction = 1 - mean_state_rmse("informed") / mean_state_rmse("open_loop")
        sensor_gains[period] = dict(zip(state_names, reduction, strict=True))
    passed = all(all(row.values()) for row in gates.values())
    report = dict(pcgym_commit=p.pcgym_revision(), casadi_version=ca.__version__, solver="CVODES",
                  fixed_reference="SciPy DOP853, independently constructed fixed-Kla PCGym model",
                  engineering_checks=gates, passed=passed, physical_checks=checks, development=runs,
                  informed_sensor_relative_state_rmse_reduction=sensor_gains,
                  endurance_plant_only=long, wall_seconds=perf_counter()-started)
    study.save(args.output, report)
    print(f"Development fixture passed={passed}, {report['wall_seconds']:.2f}s, {args.output}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
