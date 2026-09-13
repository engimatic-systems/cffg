"""Independent observation-row allocation from explicit structural intent."""

from dataclasses import dataclass, field

import numpy as np

from cffg.compiled import GaussGroupPlan


@dataclass
class ObservationBuilder:
    gauss_data: dict[str, tuple[int, int]] = field(default_factory=dict)
    n_obs_rows: int = 0
    ef_data: dict[str, tuple[int, str]] = field(default_factory=dict)
    n_ef_obs: int = 0

    def gaussian_plan(self, draft):
        obs_rows, observed_role = None, None
        if draft.observation is not None:
            observed_role = draft.observation.role_index
            rows = []
            for source in draft.observation.sources:
                if source.port is None:
                    raise KeyError(source.variable)
                self.gauss_data[source.port] = (self.n_obs_rows, source.dim)
                rows.append(self.n_obs_rows)
                self.n_obs_rows += 1
            obs_rows = np.array(rows, dtype=np.int64)
        if draft.path_problem:
            raise ValueError(draft.path_problem)
        return GaussGroupPlan(
            draft.name,
            draft.type,
            draft.roles,
            draft.dims,
            draft.members,
            draft.gather,
            draft.scatter,
            obs_rows,
            draft.kernel,
            draft.parameter_paths,
            observed_role,
        )

    def ef_rows(self, draft):
        if draft.observation is None:
            return None
        rows = []
        for source in draft.observation.sources:
            if source.port is None:
                raise KeyError(source.variable)
            self.ef_data[source.port] = (self.n_ef_obs, source.family)
            rows.append(self.n_ef_obs)
            self.n_ef_obs += 1
        return np.array(rows, dtype=np.int64)

    def gaussian_lengths(self):
        lengths = np.zeros(self.n_obs_rows, dtype=np.int64)
        for row, dim in self.gauss_data.values():
            lengths[row] = dim
        return lengths
