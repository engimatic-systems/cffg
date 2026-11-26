"""Construct, inspect and exchange complete Studio model documents."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4
from zipfile import ZipFile

import numpy as np

from cffg import Engine, compile_template, load_bundle, save_bundle
from cffg.core.ir import Template
from cffg.core.serialize import to_dict
from cffg.learn import Learnables
from cffg.models.kalman import lgssm
from cffg.models.rslds import rslds
from cffg.models.rslds_fit import vague_rslds_fit_priors
from cffg.viz import build_scene

MAX_ARCHIVE_BYTES = 256 * 1024 * 1024


def check_archive(blob):
    with ZipFile(BytesIO(blob)) as archive:
        entries = archive.infolist()
        if len(entries) > 20000 or sum(e.file_size for e in entries) > MAX_ARCHIVE_BYTES:
            raise ValueError("Bundle exceeds the import size limit")


@dataclass
class Document:
    template: Template
    arrays: dict
    learning: Learnables | None = None
    meta: dict = field(default_factory=dict)

    def bundle(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp, "model.cffgz")
            save_bundle(path, self.template, self.arrays, learnables=self.learning)
            return path.read_bytes()

    @classmethod
    def load(cls, blob, meta=None):
        check_archive(blob)
        with TemporaryDirectory() as tmp:
            path = Path(tmp, "model.cffgz")
            path.write_bytes(blob)
            loaded = load_bundle(path)
        return cls(loaded.template, loaded.arrays, loaded.learnables, meta or {})

    def sync_anchors(self):
        anchors = self.meta.setdefault("anchors", {})
        current = set()
        for kind, names in (("var", self.template.variables), ("factor", self.template.factors),
                            ("port", self.template.ports), ("blob", self.template.blobs),
                            ("plate", self.template.plates)):
            for name in names:
                ref = f"{kind}:{name}"
                current.add(ref)
                if ref not in anchors:
                    anchors[ref] = uuid4().hex
        self.meta["anchors"] = {key: value for key, value in anchors.items() if key in current}

    def describe(self):
        template = self.template
        scene = build_scene(template)
        return {
            "template": to_dict(template), "metadata": self.meta,
            "arrays": {name: {"shape": list(a.shape), "dtype": str(a.dtype), "bytes": a.nbytes,
                             "sample": a.ravel()[:8].tolist()} for name, a in self.arrays.items()},
            "scene": {"width": scene.width, "height": scene.height,
                      "items": [dict(asdict(i), primitive=type(i).__name__) for i in scene.items]},
            "resources": self.meta.get("resources", {}),
        }

    def certify(self):
        self.sync_anchors()
        data = self.bundle()
        # load_bundle checks the exact persisted arrays, learning state and compiler admission.
        verified = Document.load(data)
        from cffg.estimate import estimate
        e = estimate(Engine(compile_template(verified.template, arrays=verified.arrays)), calibrate=False)
        self.meta["resources"] = {"state_bytes": e.state_bytes, "arenas": e.arena_bytes,
                                  "parameter_bytes": e.param_bytes, "declared_work": e.declared_work}
        return data


def starter(kind, name, *, steps=3, latent=1, observed=1, regimes=2, seed=0):
    if not (1 <= steps <= 256 and 1 <= latent <= 64 and 1 <= observed <= 64 and 2 <= regimes <= 16):
        raise ValueError("Choose 1–256 steps, dimensions 1–64 and 2–16 regimes")
    if not name.strip() or len(name) > 160:
        raise ValueError("A model name must have 1–160 characters")
    learning = None
    if kind == "blank":
        t, arrays = Template(name), {}
    elif kind == "kalman":
        t, arrays = lgssm(steps, .82*np.eye(latent), .14*np.eye(latent),
                          np.eye(observed,latent), .08*np.eye(observed), np.zeros(latent), np.eye(latent))
    elif kind == "rslds":
        if steps < 2:
            raise ValueError("The rSLDS starter needs at least two time steps")
        rng = np.random.default_rng(seed)
        t, arrays = rslds(steps, np.stack([s*np.eye(latent) for s in np.linspace(.85,-.45,regimes)]),
            rng.normal(0,.15,(regimes,latent)), np.tile(.2*np.eye(latent),(regimes,1,1)),
            rng.normal(0,.65,(regimes,regimes-1,latent)), np.zeros((regimes,regimes-1)),
            np.eye(observed,latent), .12*np.eye(observed), np.zeros(latent),
            np.eye(latent), np.full(regimes,1/regimes))
        learning = Learnables(Engine(compile_template(t,arrays=arrays)),
                              priors=vague_rslds_fit_priors(regimes,latent,observed).learning_priors())
    else:
        raise ValueError(f"Unknown starter {kind!r}")
    t.name = name.strip()
    meta = {"recipe":{"kind":kind,"steps":steps,"latent":latent,"observed":observed,"regimes":regimes,"seed":seed},
            "notes":[], "reviews":[]}
    meta["axes"] = {n: ("regimes" if v.cardinality else "observed" if n in t.ports else "latent")
                    for n,v in t.variables.items()}
    return Document(t, arrays, learning, meta)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)+"\n").encode()
