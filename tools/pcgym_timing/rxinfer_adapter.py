"""Native RxInfer 3.6 controller in a persistent, single-threaded Julia worker.

The private Project.toml and Manifest.toml are in PCGYM_RXINFER_PROJECT
(default ~/agent-share/pcgym-four-library-timing/rxinfer). Reproduce with

    export PCGYM_RXINFER_PROJECT=~/agent-share/pcgym-four-library-timing/rxinfer
    julia --project="$PCGYM_RXINFER_PROJECT" -e 'using Pkg; Pkg.instantiate()'

Use Julia 1.10.11. Set PCGYM_JULIA to the Julia executable if it is outside PATH.
"""
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess

import numpy as np


class Adapter:
    def __init__(self, arrays, config, *, affine=None):
        project = Path(os.environ.get(
            "PCGYM_RXINFER_PROJECT", "~/agent-share/pcgym-four-library-timing/rxinfer"
        )).expanduser()
        manifest = project / "Manifest.toml"
        if not manifest.is_file():
            raise FileNotFoundError(f"missing pinned RxInfer manifest in {project}")
        worker = Path(__file__).with_name("rxinfer_worker.jl")
        env = {**os.environ, "JULIA_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
               "VECLIB_MAXIMUM_THREADS": "1", "JULIA_LOAD_PATH": "@:@stdlib"}
        self.proc = subprocess.Popen(
            [os.environ.get("PCGYM_JULIA", "julia"), "--startup-file=no", "--threads=1",
             f"--project={project}", str(worker)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, encoding="utf-8", bufsize=1, env=env,
        )
        try:
            result = self._request(dict(
                op="init", arrays={k: np.asarray(v).tolist() for k, v in arrays.items()},
                config=config, affine=None if affine is None else [np.asarray(x).tolist() for x in affine],
            ))
            if result["status"] != "ready":
                raise RuntimeError(result.get("error", str(result)))
            self.metadata = {k: v for k, v in result.items() if k != "status"}
            self.metadata.update(project=str(project), manifest_sha256=sha256(manifest.read_bytes()).hexdigest())
        except BaseException:
            self.close()
            raise

    def _request(self, request):
        self.proc.stdin.write(json.dumps(request, allow_nan=False) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"RxInfer worker exited with code {self.proc.wait()}")
        return json.loads(line)

    def solve(self, data, *, details=False):
        try:
            values = (data[f"e{i}"] for i in range(20))
            rows = [None if value is None else np.asarray(value).tolist() for value in values]
            return self._request(dict(op="solve", data=rows, details=details))
        except (KeyError, TypeError, ValueError, OSError, RuntimeError) as exc:
            return dict(status="error", action=None, error=str(exc))

    def close(self):
        self.proc.stdin.close()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        self.proc.stdout.close()
