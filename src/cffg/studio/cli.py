"""Command-line access to Studio for humans and coding agents."""

import argparse
import json
import os
from pathlib import Path
import sys
from uuid import uuid4

import httpx


def main():
    p = argparse.ArgumentParser(
        prog="cffg-studio",
        description="Inspect, edit, review and run versioned CFFG models.",
    )
    p.add_argument(
        "--url", default=os.getenv("CFFG_STUDIO_URL", "http://127.0.0.1:8493")
    )
    subs = p.add_subparsers(dest="command", required=True)
    subs.add_parser("models")
    subs.add_parser("capabilities")
    for name in ("show", "history", "events", "export"):
        cmd = subs.add_parser(name)
        cmd.add_argument("model")
        if name in ("show", "export"):
            cmd.add_argument("--revision", default="HEAD")
        if name == "export":
            cmd.add_argument("file", type=Path)
    c = subs.add_parser("new")
    c.add_argument("name")
    c.add_argument("--kind", choices=("blank", "kalman", "rslds"), default="blank")
    c = subs.add_parser("find")
    c.add_argument("query", nargs="?", default="")
    c = subs.add_parser("import")
    c.add_argument("file", type=Path)
    for name in ("preview", "apply", "critique", "propose"):
        c = subs.add_parser(name)
        c.add_argument("model")
        c.add_argument("--base", required=True)
        c.add_argument("--request-id", default=None)
        if name in ("preview", "apply", "propose"):
            c.add_argument(
                "--commands", default="-", help="JSON command array file or - for stdin"
            )
        if name in ("critique", "propose"):
            c.add_argument("--text", required=True)
            c.add_argument("--ref", default="")
    c = subs.add_parser("diff")
    c.add_argument("model")
    c.add_argument("base")
    c.add_argument("--to", default="HEAD")
    c = subs.add_parser("run")
    c.add_argument("model")
    c.add_argument("--revision", required=True)
    c.add_argument(
        "--kind",
        choices=("run", "profile", "trace", "simulate", "generate"),
        default="run",
    )
    c.add_argument("--seed", type=int, default=7)
    c.add_argument("--iterations", type=int, default=12)
    c.add_argument("--tape")
    c.add_argument("--prompt", default="")
    c.add_argument("--source", type=Path)
    c = subs.add_parser("job")
    c.add_argument("id")
    c.add_argument("--follow", action="store_true")
    c = subs.add_parser("token")
    c.add_argument("operation", choices=("issue", "list", "revoke"))
    c.add_argument("name", nargs="?", default="")
    c.add_argument("--scope", choices=("read", "edit"), default="edit")
    c = subs.add_parser("serve")
    c.add_argument("--port", type=int, default=8493)
    args = p.parse_args()
    home = Path(os.getenv("CFFG_STUDIO_HOME", Path.home() / ".local/share/cffg-studio"))
    if args.command == "serve":
        import uvicorn

        uvicorn.run(
            "cffg.studio.server:create_app",
            factory=True,
            host="127.0.0.1",
            port=args.port,
            proxy_headers=False,
        )
        return
    if args.command == "token":
        from .access import Access

        home.mkdir(mode=0o700, parents=True, exist_ok=True)
        print(
            json.dumps(
                Access(home).manage(args.operation, args.name, args.scope), indent=2
            )
        )
        return
    token = os.environ.get("CFFG_STUDIO_TOKEN")
    if not token and (home / "agent-token").exists():
        from urllib.parse import urlsplit

        if urlsplit(args.url).hostname not in ("127.0.0.1", "localhost", "::1"):
            p.error("Set CFFG_STUDIO_TOKEN to a scoped token for a remote URL")
        token = (home / "agent-token").read_text().strip()
    client = httpx.Client(
        base_url=args.url,
        headers={"Authorization": "Bearer " + token} if token else {},
        timeout=180,
    )
    op = args.command
    model = getattr(args, "model", "")
    path = "/v1/models/" + model

    def commands():
        return json.loads(
            sys.stdin.read()
            if args.commands == "-"
            else Path(args.commands).read_text()
        )

    try:
        if op == "models":
            res = client.get("/v1/models")
        elif op == "capabilities":
            res = client.get("/v1/capabilities")
        elif op == "new":
            res = client.post("/v1/models", json={"name": args.name, "kind": args.kind})
        elif op == "find":
            res = client.get("/v1/search", params={"q": args.query})
        elif op == "import":
            res = client.post(
                "/v1/import/bundle",
                content=args.file.read_bytes(),
                headers={"Content-Type": "application/zip"},
            )
        elif op == "show":
            res = client.get(
                path
                if args.revision == "HEAD"
                else path + "/revisions/" + args.revision
            )
        elif op == "diff":
            res = client.get(path + "/diff", params={"base": args.base, "to": args.to})
        elif op == "history":
            res = client.get(path + "/history")
        elif op == "export":
            res = client.get(path + "/export", params={"revision": args.revision})
            res.raise_for_status()
            args.file.write_bytes(res.content)
            return
        elif op in ("events", "job") and (op == "events" or args.follow):
            endpoint = path if op == "events" else "/v1/runs/" + args.id
            with client.stream("GET", endpoint + "/events", timeout=None) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        print(line[6:], flush=True)
            return
        elif op in ("preview", "apply", "critique", "propose"):
            payload = {"base": args.base, "request_id": args.request_id or uuid4().hex}
            if op != "critique":
                payload["commands"] = commands()
            if op in ("critique", "propose"):
                payload.update(text=args.text, ref=args.ref)
            route = {
                "preview": "transactions/preview",
                "apply": "transactions",
                "critique": "critiques",
                "propose": "proposals",
            }[op]
            res = client.post(path + "/" + route, json=payload)
        elif op == "run":
            res = client.post(
                "/v1/runs",
                json={
                    "model": model,
                    "revision": args.revision,
                    "kind": args.kind,
                    "seed": args.seed,
                    "iterations": args.iterations,
                    "tape": args.tape,
                    "prompt": args.prompt,
                    "source": args.source.read_text() if args.source else "",
                },
            )
        elif op == "job":
            res = client.get("/v1/runs/" + args.id)
        res.raise_for_status()
        print(json.dumps(res.json(), ensure_ascii=False, indent=2))
    except httpx.HTTPStatusError as e:
        print(e.response.text, file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
