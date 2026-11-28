"""Persistent native RxInfer slices for a goal-free filter and two-flow planner."""

from hashlib import sha256
import os
from pathlib import Path
import struct
import subprocess

import numpy as np


_PROJECT = Path.home() / "agent-share/pcgym-four-library-timing/rxinfer"
_MANIFEST_SHA256 = "21c7229c071e5445491aac7641c66854d91555c12a06ae02d96be51f8594c1c2"


def _array(value, shape, name):
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    array = np.asarray(value, dtype="<f8")
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}, got {array.shape}")
    return array


class Backend:
    def __init__(self):
        project = Path(os.environ.get("PCGYM_RXINFER_PROJECT", _PROJECT)).expanduser()
        if sha256((project / "Manifest.toml").read_bytes()).hexdigest() != _MANIFEST_SHA256:
            raise RuntimeError("Expected the reactor study's pinned RxInfer 3.6 manifest")
        julia = os.environ.get("PCGYM_JULIA", "julia")
        self._proc = subprocess.Popen(
            [julia, "--startup-file=no", "--threads=1", f"--project={project}",
             str(Path(__file__).with_suffix(".jl"))],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            env={**os.environ, "JULIA_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                 "VECLIB_MAXIMUM_THREADS": "1", "JULIA_LOAD_PATH": "@:@stdlib"},
        )
        try:
            if self._read(4) != b"RXE1":
                raise RuntimeError("Unexpected RxInfer worker handshake")
        except BaseException:
            self.close()
            raise
        self.last_innovation_nll = None

    def _read(self, size):
        data = self._proc.stdout.read(size)
        if len(data) != size:
            raise RuntimeError(f"RxInfer worker closed its output (exit {self._proc.poll()})")
        return data

    def _request(self, op, n, k, horizon, arrays):
        payload = struct.pack("<Biii", op, n, k, horizon) + b"".join(a.tobytes(order="F") for a in arrays)
        self._proc.stdin.write(struct.pack("<I", len(payload)) + payload)
        self._proc.stdin.flush()
        status, size = struct.unpack("<BI", self._read(5))
        data = self._read(size)
        if status:
            raise RuntimeError(data.decode("utf8", errors="replace"))
        return np.frombuffer(data, dtype="<f8").copy()

    def update(self, m, P, A, b, Q, H, R, y):
        """Condition the supplied state prior, with the applied input included in b."""
        self.last_innovation_nll = None
        n = len(m)
        args = [_array(m, (n,), "m"), _array(P, (n, n), "P"), _array(A, (n, n), "A"),
                _array(b, (n,), "b"), _array(Q, (n, n), "Q")]
        k = 0
        if y is not None:
            y = np.atleast_1d(y)
            k = len(y)
            if not k:
                raise ValueError("An observed reading must have at least one channel")
            args.extend((_array(H, (k, n), "H"), _array(R, (k, k), "R"), _array(y, (k,), "y")))
        reply = self._request(1, n, k, 1, args)
        self.last_innovation_nll = float(reply[-1])
        return reply[:n], reply[n:-1].reshape((n, n), order="F")

    def plan(self, m, P, As, bs, Bs, Qs, C, targets, Rgoal, U0, Ucov):
        """Return the raw posterior mean of the first two flows under all horizon preferences."""
        n, horizon = len(m), len(As)
        if not horizon:
            raise ValueError("The planning horizon must be positive")
        C = np.atleast_2d(C)
        k = len(C)
        targets = np.asarray(targets)
        if targets.shape == (horizon,) and k == 1:
            targets = targets[:, None]
        targets = _array(targets, (horizon, k), "targets")
        U0 = np.broadcast_to(U0, (horizon, 2))
        args = [_array(m, (n,), "m"), _array(P, (n, n), "P"), _array(C, (k, n), "C"),
                _array(np.atleast_2d(Rgoal), (k, k), "Rgoal"), _array(Ucov, (2, 2), "Ucov")]
        for a, b, B, q, u, target in zip(As, bs, Bs, Qs, U0, targets, strict=True):
            args.extend((_array(a, (n, n), "A"), _array(b, (n,), "b"), _array(B, (n, 2), "B"),
                         _array(q, (n, n), "Q"), _array(u, (2,), "u"), target))
        return self._request(2, n, k, horizon, args)

    def close(self):
        self._proc.stdin.close()
        try:
            self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
        self._proc.stdout.close()
