"""A portable hard-box decision over the complete D13.6 directed Gaussian chain."""

from dataclasses import asdict, dataclass, replace
import json
from weakref import WeakValueDictionary

import jax
import numpy as np

from cffg import Engine, compile_template
from cffg.bounded_actions import BoundedGaussianPlan, BoundedResult
from cffg.core import PortRole, Template
from cffg.core.compose import scoped
from cffg.core.serialize import from_dict, to_dict
from cffg.kernel_runtime import parameters
from cffg.runtime_types import RunStatus


_solve_joint = jax.jit(BoundedGaussianPlan.solve_joint, static_argnums=0)


@dataclass(frozen=True)
class BoundedCoordinate:
    edge: str
    coordinate: int
    lower: float
    upper: float


@dataclass(frozen=True)
class ChainDecision:
    """Joint order is selected actions, then remaining time-major state coordinates."""

    order: tuple[tuple[int, int], ...]
    gaussian_mean: np.ndarray
    gaussian_covariance: np.ndarray
    gaussian_free_energy: float
    bounded: BoundedResult


class _RegionEngine(Engine):
    """Keep certificate provenance without retaining completed decision windows."""

    def __init__(self, arena, declaration):
        super().__init__(arena)
        self.declaration = declaration
        self.results = WeakValueDictionary()

    def run(self, *args, **kwargs):
        result = super().run(*args, **kwargs)
        self.results[id(result)] = result
        return result


@dataclass(frozen=True)
class BoundedChainRegion:
    template: Template
    actions: tuple[BoundedCoordinate, ...]

    def __post_init__(self):
        object.__setattr__(self, "actions", tuple(self.actions))
        if len(self.actions) not in (1, 2):
            raise ValueError("a connected directed chain supports one or two bounded coordinates")
        for action in self.actions:
            if type(action.coordinate) is not int or action.coordinate < 0:
                raise ValueError("invalid bounded coordinate")
            if not np.isfinite((action.lower, action.upper)).all():
                raise ValueError("bounded coordinates require finite endpoints")

    def to_dict(self):
        return dict(kind="directed_chain_box_v1", template=to_dict(self.template),
                    actions=[asdict(a) for a in self.actions])

    @classmethod
    def from_dict(cls, document):
        if set(document) != {"kind", "template", "actions"} or document["kind"] != "directed_chain_box_v1":
            raise ValueError("unknown directed chain decision declaration")
        return cls(from_dict(document["template"]), tuple(BoundedCoordinate(**a) for a in document["actions"]))

    def scoped(self, prefix):
        return type(self)(scoped(self.template, prefix),
                          tuple(replace(a, edge=f"{prefix}/{a.edge}") for a in self.actions))

    def compile(self, *, arrays=None, root=None):
        if any(port.role != PortRole.DATA for port in self.template.ports.values()):
            raise ValueError("closed directed region admits DATA ports only")
        arena = compile_template(self.template, arrays=arrays, root=root)
        engine = _RegionEngine(arena, json.dumps(self.to_dict(), sort_keys=True))
        self._indices(engine)
        return engine

    def _indices(self, engine):
        plan = getattr(engine, "directed_chain_plan", None)
        if plan is None:
            raise ValueError("source requires a complete recognized directed Taylor chain")
        # All three legs of each equality refer to the same latent state.
        by_slot = {s: i for i, site in enumerate(plan.sites[:-1]) for s in site.equality_scatter}
        selection = []
        for a in self.actions:
            edge = engine.arena.edges.get(a.edge)
            sites = set() if edge is None else {by_slot[s] for s in (edge.slot0, edge.slot1) if s in by_slot}
            if edge is None or not edge.interior or len(sites) != 1 or a.coordinate >= edge.dim:
                raise ValueError(f"{a.edge}[{a.coordinate}] must select a nonterminal transition state")
            selection.append((sites.pop(), a.coordinate))
        if len(set(selection)) != len(selection):
            raise ValueError("duplicate or equality-aliased action coordinate")
        return tuple(selection)

    def evaluate(self, engine, result):
        """Return the conditional mean decision and its normalized frozen objective."""
        if (not isinstance(engine, _RegionEngine)
                or engine.declaration != json.dumps(self.to_dict(), sort_keys=True)
                or engine.results.get(id(result)) is not result):
            raise ValueError("region, engine or directed result identity mismatch")
        if (result.status != RunStatus.CONVERGED or result.failure is not None
                or result.termination_reason.value != "directed_chain_certified"
                or len(result.free_energies) != 1 or not np.isfinite(result.residual)):
            raise ValueError("decision requires a certified terminal directed result")
        selection = self._indices(engine)
        chain = engine._directed
        if not chain.admits(result.state):
            raise ValueError("result has unsupported exterior messages or bindings")
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            mean, cov, energy = _frozen_joint(chain, result.state)
        # Recheck the terminal state's ledger and cavities with the reconstructed
        # evidence. State.params is a mapping and may have been edited after run.
        tol = max(1e-6, 64 * np.finfo(mean.dtype).eps)
        check = engine._certify_directed(result.state, energy, tol)
        if not check.passed or abs(check.free_energy-result.free_energies[0]) > tol*(1+abs(energy)):
            raise ValueError("terminal certificate disagrees with live bindings or frozen evidence")
        dim = chain.plan.dim
        order = selection + tuple((t, i) for t in range(len(chain.plan.sites)) for i in range(dim)
                                  if (t, i) not in selection)
        indices = [t * dim + i for t, i in order]
        mean, cov = mean[indices], cov[np.ix_(indices, indices)]
        box = BoundedGaussianPlan((len(selection),), tuple(a.lower for a in self.actions),
                                  tuple(a.upper for a in self.actions))
        bounded = _solve_joint(box, mean, cov, energy)
        if bounded.mean.dtype != mean.dtype:
            raise FloatingPointError("bounded solve changed the certified chain dtype")
        return ChainDecision(order, mean, cov, energy, bounded)


def _frozen_joint(chain, state):
    """Condition the growing trajectory at every present DATA row.

    Appending A*x+b and independent noise gives Cov(history, next) =
    Cov(history, last) A.T. Each innovation density contributes once to F_G.
    """
    plan, arena = chain.plan, chain.arena
    dim = plan.dim
    dtype = state.g_eta.dtype
    live = {k: np.asarray(v) for k, v in state.params.items()}
    arrays = (state.g_lam, state.g_mask, state.g_obs, *live.values())
    if dtype not in (np.float32, np.float64) or any(v.dtype != dtype for v in arrays):
        raise ValueError("directed decision requires uniform certified floating dtype")

    def row(member):
        return parameters(member.group, live, member.row)

    prior = row(plan.prior)
    mean, cov = prior["m"], np.linalg.inv(prior["W"])
    energy = dtype.type(0)
    for site in plan.sites:
        obs = row(site.emission)
        if state.g_mask[site.obs_row]:
            h, offset = obs["A"], obs["b"]
            value = np.asarray(state.g_obs)[arena.g_obs_indices(site.obs_row, len(offset))]
            residual = value - (h @ mean[-dim:] + offset)
            noise = np.linalg.inv(obs["W"])
            cross = cov[:, -dim:] @ h.T
            innovation = noise + h @ cross[-dim:]
            chol = np.linalg.cholesky(innovation)
            whitened = np.linalg.solve(chol, residual)
            energy += (len(value)*np.log(dtype.type(2*np.pi)) + 2*np.log(chol.diagonal()).sum()
                       + whitened @ whitened) / 2
            gain = np.linalg.solve(innovation, cross.T).T
            mean = mean + gain @ residual
            transform = np.eye(len(mean), dtype=dtype)
            transform[:, -dim:] -= gain @ h
            cov = transform @ cov @ transform.T + gain @ noise @ gain.T
        if site.transition is not None:
            ref = site.transition
            slot = ref.gather[1]
            precision = np.asarray(state.g_lam)[arena.g_lam_indices(slot, dim)]
            eta = np.asarray(state.g_eta)[arena.g_eta_indices(slot, dim)]
            center = np.linalg.solve(precision, eta)
            params = row(ref)
            matrix, value = chain.linearize(ref.group.members[ref.row], center, params["theta"])
            if matrix.dtype != dtype or value.dtype != dtype:
                raise ValueError("Taylor binding changed the certified chain dtype")
            nxt = value + matrix @ (mean[-dim:] - center)
            cross = cov[:, -dim:] @ matrix.T
            predicted = matrix @ cross[-dim:] + np.linalg.inv(params["W"])
            mean = np.concatenate((mean, nxt))
            cov = np.block([[cov, cross], [cross.T, predicted]])
    return mean, cov, energy
