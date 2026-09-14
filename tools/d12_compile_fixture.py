"""Developer-only moderate-chain fixture and output selection; no solver mathematics."""


def fixture_arrays(horizon):
    """Evaluate the frozen decimal formulas, then store native float32 inputs."""
    import numpy as np

    if horizon not in (32, 100):
        raise ValueError('the compilation plan contains only T=32 and T=100')
    i, stick = np.arange(4)[:, None], np.arange(3)[None, :]
    weights = np.stack(np.broadcast_arrays(.12 * (i - 1.5) + .04 * (stick - 1),
                                           -.08 * (i - 1.5) + .03 * (stick - 1)), axis=-1)
    offsets = .10 * (i - 1.5) - .15 * (stick - 1)
    matrix = np.array([[[.65 + .04 * j, .03], [-.02, .80 - .03 * j]] for j in range(4)])
    bias = np.array([[.10 - .05 * j, -.15 + .04 * j] for j in range(4)])
    noise = np.array([np.diag([.8 + .1 * j, 1.1 + .05 * j]) for j in range(4)])
    t, i = np.arange(horizon)[:, None], np.arange(4)[None, :]
    constant = np.broadcast_to(-.2 - .1 * i, (horizon, 4))
    linear = np.stack(np.broadcast_arrays(.1 * (i - 1.5) + .05 * (-1.) ** t,
                                          -.08 * (i - 1.5) + .03 * ((t % 3) - 1)), axis=-1)
    precision = np.broadcast_to(np.array([np.diag([.9 + .1 * j, 1.1 + .05 * j])
                                         for j in range(4)]), (horizon, 4, 2, 2))
    transition = dict(weights=weights, offsets=offsets, xi=np.zeros_like(offsets),
                      matrix=matrix, bias=bias, noise=noise)
    parameters = {name: np.broadcast_to(value, (horizon - 1, *value.shape)).astype(np.float32)
                  for name, value in transition.items()}
    sites = dict(constant=constant.astype(np.float32), linear=linear.astype(np.float32),
                 precision=precision.astype(np.float32), status=np.zeros(horizon, np.int32),
                 center=np.zeros((horizon, 4, 2), np.float32))
    return parameters, sites


def context(horizon):
    """Build fixed layout only. Never initialize, evaluate or solve D12 here."""
    import jax
    from cffg.hybrid_joint import Coordinate
    from cffg.hybrid_natural import Natural
    from cffg.hybrid_separator import TransitionParameters

    parameters, sites = fixture_arrays(horizon)
    parameters = TransitionParameters(**jax.tree.map(jax.numpy.asarray, parameters))
    sites = Natural(**jax.tree.map(jax.numpy.asarray, sites))
    return Coordinate.from_sites(sites), parameters, sites


def inference_outputs(state):
    """Select returned fields without changing evaluation or failure dependencies."""
    evaluation = state.evaluation
    result = dict(coordinate=state.coordinate, free_energy=evaluation.local.free_energy,
                  dual_energy=evaluation.local.dual_energy, certificates=evaluation.certificates,
                  evaluation_status=evaluation.status, local_status=evaluation.local.status,
                  auxiliary_status=evaluation.auxiliary.status)
    for name in ('iterations', 'evaluations', 'rejected_trials', 'linear_iterations',
                 'gauss_newton_steps', 'relative_energy_change', 'rejection_counts',
                 'last_rejection', 'status', 'reason'):
        result[name] = getattr(state, name)
    return result
