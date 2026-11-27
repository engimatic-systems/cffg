"""Install a private Studio release on this Mac and publish its agent-share launcher."""

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import tempfile
from urllib.request import Request, urlopen
import time


def run(*args, **kwargs):
    return subprocess.run([str(x) for x in args], check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--login",
        required=True,
        help="Tailscale user login allowed to open the workstation",
    )
    parser.add_argument("--backend-port", type=int, default=8493)
    parser.add_argument(
        "--https-port", type=int, default=8098,
        help="Private HTTPS port allowed from the remote clients (default: 8098)",
    )
    parser.add_argument("--share", type=Path, default=Path.home() / "agent-share")
    args = parser.parse_args()
    src = Path(__file__).resolve().parents[1]
    home = Path.home() / ".local/share/cffg-studio"
    home.mkdir(mode=0o700, parents=True, exist_ok=True)
    uv = shutil.which("uv")
    tailscale = (
        shutil.which("tailscale")
        or "/Applications/Tailscale.app/Contents/MacOS/Tailscale"
    )
    status = json.loads(run(tailscale, "status", "--json", capture_output=True).stdout)
    if status["BackendState"] != "Running":
        raise SystemExit("Start Tailscale before installing Studio")
    host = status["Self"]["DNSName"].rstrip(".")
    target = f"http://127.0.0.1:{args.backend_port}"
    authority = f"{host}:{args.https_port}"
    serve = json.loads(
        run(tailscale, "serve", "status", "--json", capture_output=True).stdout or "{}"
    )
    existing = serve.get("Web", {}).get(authority, {}).get("Handlers", {})
    if existing and existing != {"/": {"Proxy": target}}:
        raise SystemExit(f"A different Tailscale handler already owns {authority}")
    with tempfile.TemporaryDirectory() as tmp:
        run(uv, "build", "--wheel", "--out-dir", tmp, cwd=src)
        wheel = next(Path(tmp).glob("*.whl"))
        release = (
            home / "releases" / hashlib.sha256(wheel.read_bytes()).hexdigest()[:16]
        )
        release.mkdir(parents=True, exist_ok=True)
        python = release / "venv/bin/python"
        if not python.exists():
            run(uv, "venv", release / "venv", "--python", os.sys.executable)
        requirements = Path(tmp) / "requirements.txt"
        run(
            uv,
            "export",
            "--locked",
            "--extra",
            "studio",
            "--no-default-groups",
            "--no-emit-project",
            "--format",
            "requirements-txt",
            "--output-file",
            requirements,
            cwd=src,
            stdout=subprocess.DEVNULL,
        )
        run(
            uv,
            "pip",
            "install",
            "--python",
            python,
            "--require-hashes",
            "-r",
            requirements,
        )
        run(uv, "pip", "install", "--python", python, "--no-deps", wheel)
    run(
        python,
        "-c",
        "from cffg.studio.server import create_app; from cffg.studio.cli import main",
        cwd=release,
    )
    # Supply examples on the first installation without touching previously authored models.
    init_script = """from pathlib import Path
from cffg.studio.library import Library
from cffg.studio.access import Access
from cffg.studio.models import starter
Access(Path.home()/'.local/share/cffg-studio')
lib=Library(Path.home()/'projects/cffg-models')
if not lib.list():
    for kind,name in [('rslds','Switching dynamics'),('kalman','Linear Gaussian tracker'),('blank','Blank graph')]:
        lib.create(starter(kind,name),'Studio templates')
"""
    run(python, "-c", init_script, cwd=release)
    label = "com.ravd.cffg-studio"
    plist = Path.home() / "Library/LaunchAgents" / (label + ".plist")
    plist.parent.mkdir(exist_ok=True, parents=True)
    env = {
        "PATH": str(home / "bin")
        + ":/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
        "CFFG_STUDIO_ENGINE_REVISION": release.name,
        "CFFG_STUDIO_HOST": host,
        "CFFG_STUDIO_LOGIN": args.login,
        "HOME": str(Path.home()),
    }
    config = {
        "Label": label,
        "ProgramArguments": [
            str(python),
            "-m",
            "cffg.studio.cli",
            "serve",
            "--port",
            str(args.backend_port),
        ],
        "WorkingDirectory": str(release),
        "EnvironmentVariables": env,
        "RunAtLoad": True,
        "KeepAlive": True,
        "ThrottleInterval": 10,
        "StandardOutPath": str(home / "server.log"),
        "StandardErrorPath": str(home / "server-error.log"),
    }
    domain = f"gui/{os.getuid()}"
    old = plist.read_bytes() if plist.exists() else None

    def restart():
        subprocess.run(
            ["launchctl", "bootout", domain + "/" + label], capture_output=True
        )
        # launchd can acknowledge bootout before the previous job has retired.
        deadline = time.monotonic() + 45
        while True:
            attempt = subprocess.run(
                ["launchctl", "bootstrap", domain, str(plist)],
                capture_output=True,
                text=True,
            )
            if attempt.returncode == 0:
                break
            if attempt.returncode != 5 or time.monotonic() >= deadline:
                raise RuntimeError(
                    f"launchd could not start Studio: {attempt.stderr.strip()}"
                )
            time.sleep(0.5)

    try:
        plist.write_bytes(plistlib.dumps(config))
        restart()
        for _ in range(60):
            try:
                req = Request(
                    target + "/v1/capabilities",
                    headers={
                        "Authorization": "Bearer "
                        + (home / "agent-token").read_text().strip()
                    },
                )
                with urlopen(req, timeout=2) as res:
                    if res.status == 200:
                        break
            except OSError:
                time.sleep(0.5)
        else:
            raise RuntimeError("Studio failed its local health check")
        run(tailscale, "serve", "--bg", f"--https={args.https_port}", target)
    except BaseException:
        if old is not None:
            plist.write_bytes(old)
            restart()
        raise
    url = "https://" + authority + "/"
    relative = args.share / "cffg-studio/index.html"
    relative.parent.mkdir(parents=True, exist_ok=True)
    escaped = html.escape(url, quote=True)
    page = (
        f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        f'<title>CFFG Studio</title><meta http-equiv="refresh" content="0; url={escaped}"><body>'
        f'<a href="{escaped}">Open CFFG Studio</a></body></html>'
    )
    temporary = relative.with_suffix(".html.tmp")
    temporary.write_text(page)
    os.replace(temporary, relative)
    print(f"Studio is ready at {url}\nLauncher: {relative}\nRelease: {release}")


if __name__ == "__main__":
    main()
