"""recurrence readouts over explicit structure and current numerical inputs."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from cffg import recurrence
from cffg.execution.support import log_support as _jax_log_support
from cffg.readouts.types import (
    PairBeliefs,
)
from cffg.readouts.views import (
    RecurrenceView,
)
from cffg.runtime_types import State


def recurrence_pair_beliefs(
    view: RecurrenceView, state: State
) -> dict[str, dict[str, np.ndarray]]:
    """Factor name -> normalized structured recurrence pair belief.

    Pair rows are source regimes (the ``in`` role) and columns are
    destination regimes (the ``out`` role). ``pair_support`` records
    literal categorical support independently of floating-point
    probability underflow. The readout also exposes the direct Gaussian
    drive moments and D9.12 masses shared by inference and learning.
    Improper inputs preserve support but return shape-correct NaNs and
    ``proper=False`` for numeric fields.
    """
    out: dict[str, dict[str, np.ndarray]] = {}
    for group in view.recurrence_groups:
        card = group.cards[0]
        destination = state.messages[
            view.message_indices(group.d_gather_both[:, 0, 1], card)
        ]
        source = state.messages[
            view.message_indices(group.d_gather_both[:, 1, 1], card)
        ]
        destination = jnp.where(_jax_log_support(destination), destination, -jnp.inf)
        source = jnp.where(_jax_log_support(source), source, -jnp.inf)
        drive_eta = state.g_eta[
            view.g_eta_indices(group.drive_gather_both, group.dim)
        ].sum(axis=1)
        drive_precision = state.g_lam[
            view.g_lam_indices(group.drive_gather_both, group.dim)
        ].sum(axis=1)
        beliefs = recurrence.structured_readout(
            destination,
            source,
            drive_eta,
            drive_precision,
            state.params[group.name + ":weights"],
            state.params[group.name + ":offsets"],
        )
        for index, name in enumerate(group.members):
            out[name] = {
                "pair_probability": np.asarray(beliefs.pair_probability[index]),
                "source_probability": np.asarray(beliefs.source_probability[index]),
                "destination_probability": np.asarray(
                    beliefs.destination_probability[index]
                ),
                "pair_support": np.asarray(beliefs.pair_support[index]),
                "source_support": np.asarray(beliefs.source_support[index]),
                "destination_support": np.asarray(beliefs.destination_support[index]),
                "drive_mean": np.asarray(beliefs.drive_mean[index]),
                "drive_covariance": np.asarray(beliefs.drive_covariance[index]),
                "drive_second": np.asarray(beliefs.drive_second[index]),
                "omega": np.asarray(beliefs.omega[index]),
                "exposure": np.asarray(beliefs.exposure[index]),
                "signed_success": np.asarray(beliefs.signed_success[index]),
                "proper": np.asarray(beliefs.proper[index]),
            }
    return out


def recurrence_weight_stats(
    view: RecurrenceView, state: State, *, pair_beliefs: PairBeliefs
) -> dict[str, dict[str, np.ndarray]]:
    """Factor-local PG statistics for between-trial recurrence learning.

    The result is keyed by ``recurrence_link`` factor name. ``linear``
    and ``quadratic`` are pair-derived natural-parameter increments for
    independent Gaussian posteriors over each augmented coefficient
    ``[R_il; r_il]``. Scalar predicate and tolerance fields explain an
    improper row.
    """
    out: dict[str, dict[str, np.ndarray]] = {}
    beliefs = pair_beliefs(state)
    for group in view.recurrence_groups:
        rows = [beliefs[name] for name in group.members]
        stats = recurrence.weight_stats(
            jnp.stack([jnp.asarray(row["pair_probability"]) for row in rows]),
            jnp.stack([jnp.asarray(row["drive_mean"]) for row in rows]),
            jnp.stack([jnp.asarray(row["drive_covariance"]) for row in rows]),
            state.params[group.name + ":weights"],
            state.params[group.name + ":offsets"],
        )
        for index, name in enumerate(group.members):
            out[name] = {
                "omega": np.asarray(stats.omega[index]),
                "exposure": np.asarray(stats.exposure[index]),
                "signed_success": np.asarray(stats.signed_success[index]),
                "linear": np.asarray(stats.linear[index]),
                "quadratic": np.asarray(stats.quadratic[index]),
                "proper": np.asarray(stats.proper[index]),
                "probability_ok": np.asarray(stats.probability_ok[index]),
                "drive_ok": np.asarray(stats.drive_ok[index]),
                "parameter_ok": np.asarray(stats.parameter_ok[index]),
                "second_ok": np.asarray(stats.second_ok[index]),
                "finite_output_ok": np.asarray(stats.finite_output_ok[index]),
                "probability_minimum": np.asarray(stats.probability_minimum[index]),
                "probability_sum_error": np.asarray(stats.probability_sum_error[index]),
                "probability_tolerance": np.asarray(stats.probability_tolerance[index]),
                "drive_min_eigenvalue": np.asarray(stats.drive_min_eigenvalue[index]),
                "drive_symmetry_error": np.asarray(stats.drive_symmetry_error[index]),
                "drive_tolerance": np.asarray(stats.drive_tolerance[index]),
                "logit_second_minimum": np.asarray(stats.logit_second_minimum[index]),
                "logit_second_tolerance": np.asarray(
                    stats.logit_second_tolerance[index]
                ),
            }
    return out
