"""Run every pre-freeze qualification and bind its output to the program and runtime."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from campaign import identity, runtime
from study import DEFAULT_ROOT, HERE, LIBRARIES, REPO, read, save, sha


def qualify(root):
    if (root/"freeze.json").exists(): raise ValueError("Campaign already frozen")
    folder=root/"qual"; folder.mkdir(parents=True,exist_ok=True)
    names=("qual_bundle.py","qualify.py","plant_check.py","dynamics_check.py","cffg_native_check.py")
    checks={str((HERE/n).relative_to(REPO)):sha(HERE/n) for n in names}
    signature=identity(runtime())
    for script,name in (("plant_check.py","plant-qualification.json"),("dynamics_check.py","nonlinear.json")):
        subprocess.run([sys.executable,str(HERE/script),"--output",str(folder/name)],check=True)
    for lib in LIBRARIES:
        subprocess.run([sys.executable,str(HERE/"qualify.py"),lib,"--output",str(root)],
                       env=dict(os.environ,EXTRACTION_LAUNCH=str(time.time()),JAX_ENABLE_COMPILATION_CACHE="false"),
                       check=True)
    if identity(runtime())!=signature or any(sha(REPO/k)!=v for k,v in checks.items()):
        raise ValueError("Source changed during qualification")
    outputs=("plant-qualification.json","nonlinear.json","cffg.json","rxinfer.json")
    if not all(read(folder/n)["passed"] for n in outputs): raise ValueError("Qualification failed")
    save(folder/"attestations.json",dict(runtime_sha256=signature,checks=checks,
                                         results={n:sha(folder/n) for n in outputs}))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=DEFAULT_ROOT)
    qualify(p.parse_args().output.resolve())
