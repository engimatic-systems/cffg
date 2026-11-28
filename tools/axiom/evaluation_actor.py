"""Compatibility entry point for the frozen Gameworld evaluation factory."""

from cffg.axiom_agent import AxiomAgent, _chunk, _context, _labels, _plan
from cffg.experiments.axiom_observation import SceneSettings
import numpy as np


class Actor(AxiomAgent):
    def __init__(self, initial_rgb, game, actions, seed, *, audit_plan=False):
        count = int(actions) if isinstance(actions, (int, np.integer)) else len(actions)
        super().__init__(initial_rgb, game, count, seed, settings=SceneSettings(count), audit_plan=audit_plan)
