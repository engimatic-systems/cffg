"""Portable nonlinear mean identity and immutable JSON configuration."""

from dataclasses import dataclass
import json
import re


@dataclass(frozen=True, init=False)
class FunctionSpec:
    impl_id: str
    version: str
    method: str
    configuration_json: str

    def __init__(self, impl_id, version, *, method="linearization", configuration=None):
        if not isinstance(impl_id, str) or not re.fullmatch(r"[\w-]+(?:\.[\w-]+)+", impl_id):
            raise ValueError("mean implementation identity must be namespaced")
        if not isinstance(version, str) or not re.fullmatch(r"\d+\.\d+\.\d+", version):
            raise ValueError("mean implementation version must be major.minor.patch")
        if method not in {"linearization", "sigma_point"}:
            raise ValueError(f"unknown nonlinear Gaussian method {method!r}")
        configuration = {} if configuration is None else configuration
        if not isinstance(configuration, dict):
            raise ValueError("mean configuration must be a JSON object")
        raw = json.dumps(configuration, sort_keys=True, separators=(",", ":"), allow_nan=False)
        for name, value in (("impl_id", impl_id), ("version", version),
                            ("method", method), ("configuration_json", raw)):
            object.__setattr__(self, name, value)

    @property
    def configuration(self):
        return json.loads(self.configuration_json)

    def to_dict(self):
        return {"impl_id": self.impl_id, "version": self.version,
                "method": self.method, "configuration": self.configuration}

    @classmethod
    def from_dict(cls, value):
        if set(value) != {"impl_id", "version", "method", "configuration"}:
            raise ValueError("mean function requires identity, version, method and configuration")
        return cls(value["impl_id"], value["version"], method=value["method"],
                   configuration=value["configuration"])

    @property
    def grouping_key(self):
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
