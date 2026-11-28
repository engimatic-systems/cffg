"""Construct, inspect and exchange complete Studio model documents."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from io import BytesIO
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np

from cffg import Engine, compile_template, load_bundle, save_bundle
from cffg.core.ir import Template
from cffg.core.serialize import to_dict
from cffg.learn import Learnables
from cffg.models.kalman import lgssm
from cffg.models.rslds import rslds
from cffg.models.rslds_fit import vague_rslds_fit_priors
from cffg.viz import LayoutStyle, build_scene
from cffg.viz.layout import OccupiedCell
from cffg.viz.scene import Label, Path as ScenePath, Terminal

MAX_ARCHIVE_BYTES = 256 * 1024 * 1024


def _describe_scene_item(item):
    record = dict(asdict(item), primitive=type(item).__name__)
    if isinstance(item, (ScenePath, Terminal)) and (label := Label.for_item(item)):
        record.update(
            label_bounds=label.bounds,
            label_font_role=label.font_role,
            label_anchor=label.anchor,
        )
    return record


def check_archive(blob):
    with ZipFile(BytesIO(blob)) as archive:
        entries = archive.infolist()
        if (
            len(entries) > 20000
            or sum(e.file_size for e in entries) > MAX_ARCHIVE_BYTES
        ):
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
        doc = cls(loaded.template, loaded.arrays, loaded.learnables, meta or {})
        if not doc.meta.get("recipe"):
            doc.recover_recipe()
        return doc

    def recover_recipe(self):
        """Identify the builder's tied parameter roles when a portable graph lacks Studio metadata."""
        t = self.template
        a = self.arrays
        is_switching = (
            {"initial_regime", "dynamics", "recurrence", "emission"}
            == t.learnables.keys()
            and {"dynamics_matrix", "emission_matrix", "initial_regime_probability"}
            <= a.keys()
            and any(f.type == "recurrence_link" for f in t.factors.values())
        )
        is_kalman = {"F", "Q", "H", "R", "m0", "P0", "zn", "zm"} == a.keys()
        if not is_switching and not is_kalman:
            return
        obs = {p.variable for p in t.ports.values() if p.role.value == "data"}
        h = a["emission_matrix" if is_switching else "H"]
        if h.ndim != 2 or not obs:
            return
        self.meta["recipe"] = {
            "kind": "rslds" if is_switching else "kalman",
            "steps": len(obs),
            "observed": h.shape[0],
            "latent": h.shape[1],
            "regimes": len(a["initial_regime_probability"]) if is_switching else 2,
            "seed": 0,
        }
        self.meta["axes"] = {
            v.name: (
                "regimes"
                if v.cardinality
                else "observed"
                if v.name in obs
                else "latent"
            )
            for v in t.variables.values()
        }

    @classmethod
    def exchange(cls, blob):
        check_archive(blob)
        with ZipFile(BytesIO(blob)) as z:
            if "model.cffgz" not in z.namelist():
                return cls.load(blob)
            if sorted(z.namelist()) != ["model.cffgz", "studio.json"]:
                raise ValueError("Invalid Studio project archive")
            meta = json.loads(z.read("studio.json"))
            data = z.read("model.cffgz")
            if meta.get("format") != "cffg-studio/1" or sha256(
                data
            ).hexdigest() != meta.get("bundle_sha256"):
                raise ValueError("Studio project manifest or bundle digest mismatch")
            document = cls.load(data)
            document.meta = {
                k: meta[k]
                for k in (
                    "axes",
                    "recipe",
                    "notes",
                    "reviews",
                    "anchors",
                    "layout",
                    "layout_version",
                    "scopes",
                    "tours",
                )
                if k in meta
            }
            if "tours" in document.meta:
                from .tours import check_portable_tours

                check_portable_tours(document.meta)
            return document

    def project(self, bundle):
        out = BytesIO()
        portable = {
            k: self.meta[k]
            for k in (
                "axes",
                "recipe",
                "notes",
                "reviews",
                "anchors",
                "layout",
                "layout_version",
                "scopes",
                "tours",
            )
            if k in self.meta
        }
        portable.update(
            format="cffg-studio/1", bundle_sha256=sha256(bundle).hexdigest()
        )
        with ZipFile(out, "w", ZIP_DEFLATED) as archive:
            archive.writestr("model.cffgz", bundle)
            archive.writestr("studio.json", json_bytes(portable))
        return out.getvalue()

    def fragment(self, name, index=None):
        from cffg.core.plates import _prefix

        plate = self.template.plates[name]
        prefix = _prefix(name, index or (0,) * len(plate.dims)) + "/"
        arrays = {
            blob: self.arrays[prefix + blob if blob in plate.per_copy_blobs else blob]
            for blob in plate.fragment.blobs
        }
        metadata = self.meta.setdefault("scopes", {}).setdefault(name, {})
        return Document(plate.fragment, arrays, None, metadata)

    def sync_anchors(self):
        anchors = self.meta.setdefault("anchors", {})
        current = set()
        for kind, names in (
            ("var", self.template.variables),
            ("factor", self.template.factors),
            ("port", self.template.ports),
            ("blob", self.template.blobs),
            ("plate", self.template.plates),
        ):
            for name in names:
                ref = f"{kind}:{name}"
                current.add(ref)
                if ref not in anchors:
                    anchors[ref] = uuid4().hex
        self.meta["anchors"] = {
            key: value for key, value in anchors.items() if key in current
        }

        for name in self.template.plates:
            self.fragment(name).sync_anchors()

    def layout_scene(self):
        offsets = self.meta.get("layout")
        version = self.meta.get("layout_version")
        if version != 1:
            try:
                return build_scene(self.template, offsets=offsets), 2
            except OccupiedCell:
                if version is not None or not offsets:
                    raise
        # A preexisting offset can point to an occupied cell on the new grid.
        scene = build_scene(
            self.template, style=LayoutStyle(temporal_flow=False), offsets=offsets
        )
        return scene, 1

    def describe(self):
        template = self.template
        scene, _ = self.layout_scene()
        return {
            "template": to_dict(template),
            "metadata": self.meta,
            "arrays": {
                name: {
                    "shape": list(a.shape),
                    "dtype": str(a.dtype),
                    "bytes": a.nbytes,
                    "sample": a.ravel()[:8].tolist(),
                    "sha256": sha256(a.tobytes()).hexdigest(),
                }
                for name, a in self.arrays.items()
            },
            "learning": {
                name: {
                    key: {
                        k: sha256(np.asarray(v).tobytes()).hexdigest()
                        for k, v in states[name].items()
                    }
                    for key, states in (
                        ("prior", self.learning.initial_params),
                        ("posterior", self.learning.params),
                    )
                }
                for name in self.learning.params
            }
            if self.learning
            else {},
            "scene": {
                "width": scene.width,
                "height": scene.height,
                "items": [_describe_scene_item(item) for item in scene.items],
            },
            "resources": self.meta.get("resources", {}),
        }

    def certify(self):
        from cffg.core.plates import expand_plates
        from cffg.resolution.sources import resolve_arrays

        expanded = expand_plates(self.template)
        if expanded.blobs.keys() - self.arrays.keys():
            self.arrays = {**resolve_arrays(expanded, self.arrays, None), **self.arrays}
        self.sync_anchors()
        data = self.bundle()
        # load_bundle checks the exact persisted arrays, learning state and compiler admission.
        verified = Document.load(data)
        with ZipFile(BytesIO(data)) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            identity = [
                json.loads(archive.read("model.json")),
                manifest["blob_arrays"],
                manifest["learning"],
            ]
            self.meta["model_digest"] = sha256(json_bytes(identity)).hexdigest()
        from cffg.resource_analysis import state_storage

        engine = Engine(compile_template(verified.template, arrays=verified.arrays))
        resources = state_storage(engine.init_state())
        resources.update(
            declared_array_bytes=sum(a.nbytes for a in self.arrays.values()),
            plates={name: list(p.dims) for name, p in self.template.plates.items()},
            schedule=engine.schedule.kind,
            declared_work={
                g.name: {"members": len(g.members), **g.kernel.binding.work}
                for g in engine.arena.gauss_groups
                if g.kernel and g.kernel.binding.work
            },
        )
        self.meta["resources"] = resources
        return data


def starter(kind, name, *, steps=3, latent=1, observed=1, regimes=2, seed=0):
    if not (
        1 <= steps <= 256
        and 1 <= latent <= 64
        and 1 <= observed <= 64
        and 2 <= regimes <= 16
    ):
        raise ValueError("Choose 1–256 steps, dimensions 1–64 and 2–16 regimes")
    if not name.strip() or len(name) > 160:
        raise ValueError("A model name must have 1–160 characters")
    learning = None
    if kind == "blank":
        t, arrays = Template(name), {}
    elif kind == "kalman":
        t, arrays = lgssm(
            steps,
            0.82 * np.eye(latent),
            0.14 * np.eye(latent),
            np.eye(observed, latent),
            0.08 * np.eye(observed),
            np.zeros(latent),
            np.eye(latent),
        )
    elif kind == "rslds":
        if steps < 2:
            raise ValueError("The rSLDS starter needs at least two time steps")
        rng = np.random.default_rng(seed)
        t, arrays = rslds(
            steps,
            np.stack([s * np.eye(latent) for s in np.linspace(0.85, -0.45, regimes)]),
            rng.normal(0, 0.15, (regimes, latent)),
            np.tile(0.2 * np.eye(latent), (regimes, 1, 1)),
            rng.normal(0, 0.65, (regimes, regimes - 1, latent)),
            np.zeros((regimes, regimes - 1)),
            np.eye(observed, latent),
            0.12 * np.eye(observed),
            np.zeros(latent),
            np.eye(latent),
            np.full(regimes, 1 / regimes),
        )
        learning = Learnables(
            Engine(compile_template(t, arrays=arrays)),
            priors=vague_rslds_fit_priors(regimes, latent, observed).learning_priors(),
        )
    else:
        raise ValueError(f"Unknown starter {kind!r}")
    t.name = name.strip()
    meta = {
        "recipe": {
            "kind": kind,
            "steps": steps,
            "latent": latent,
            "observed": observed,
            "regimes": regimes,
            "seed": seed,
        },
        "notes": [],
        "reviews": [],
    }
    meta["axes"] = {
        n: ("regimes" if v.cardinality else "observed" if n in t.ports else "latent")
        for n, v in t.variables.items()
    }
    return Document(t, arrays, learning, meta)


def json_bytes(value):
    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
        + "\n"
    ).encode()
