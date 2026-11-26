"""Owner and agent identities for a loopback service behind Tailscale Serve."""

import fcntl
from hashlib import sha256
import json
import os
import secrets
from uuid import uuid4

from .jobs import write


class Access:
    def __init__(self, home):
        self.home = home
        self.path = home / "tokens.json"
        self.lock = home / "tokens.lock"
        self.admin = home / "agent-token"
        if not self.admin.exists():
            with self.lock.open("a") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                if not self.admin.exists():
                    self.admin.write_text(secrets.token_urlsafe(40))
                    self.admin.chmod(0o600)

    def entries(self):
        return json.loads(self.path.read_text()) if self.path.exists() else {}

    def manage(self, action, name="", scope="edit"):
        with self.lock.open("a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            records = self.entries()
            if action == "list":
                return [
                    {k: v for k, v in row.items() if k != "digest"}
                    for row in records.values()
                ]
            if action == "issue":
                if scope not in ("read", "edit") or not name.strip():
                    raise ValueError("Supply an agent name and read or edit scope")
                token = secrets.token_urlsafe(40)
                key = uuid4().hex
                records[key] = {
                    "id": key,
                    "name": name,
                    "scope": scope,
                    "digest": sha256(token.encode()).hexdigest(),
                }
                result = {"id": key, "token": token, "scope": scope}
            elif action == "revoke":
                del records[name]
                result = {"revoked": name}
            else:
                raise ValueError("Unknown token operation")
            write(self.path, records)
            self.path.chmod(0o600)
            return result

    def identify(self, request):
        auth = request.headers.get("authorization", "")
        if auth.startswith("Bearer "):
            token = auth[7:]
            if secrets.compare_digest(token, self.admin.read_text().strip()):
                return "local owner", "owner"
            value = sha256(token.encode()).hexdigest()
            for record in self.entries().values():
                if secrets.compare_digest(value, record["digest"]):
                    return record["name"], record["scope"]
        login = request.headers.get("tailscale-user-login", "")
        owner = os.getenv("CFFG_STUDIO_LOGIN", "")
        if (
            owner
            and login
            and login.casefold() == owner.casefold()
            and request.client.host in ("127.0.0.1", "::1")
        ):
            return login, "owner"
        return None
