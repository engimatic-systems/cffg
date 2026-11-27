"""Check Option-hover in Studio's real shell with an empty, read-only test library.

Run with `uv run --with playwright python tools/check_studio_help.py`.
Install Chromium once with `uv run --with playwright playwright install chromium`.
"""

import json
import mimetypes
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import expect, sync_playwright


STATIC = Path(__file__).resolve().parents[1] / "src/cffg/studio/static"


def serve(route):
    path = urlparse(route.request.url).path
    if path in ("/v1/models", "/v1/capabilities"):
        route.fulfill(json=[] if path == "/v1/models" else {"factors": {}})
    elif path == "/" or path.startswith("/static/"):
        asset = STATIC / ("index.html" if path == "/" else Path(path).name)
        route.fulfill(path=asset, content_type=mimetypes.guess_type(asset)[0])
    else:
        route.abort()


def main():
    catalog = json.loads((STATIC / "tooltips.json").read_text())["entries"]
    assert all(row["review"] == "approved" and row["text"] for row in catalog.values())
    with sync_playwright() as pw, pw.chromium.launch() as browser:
        page = browser.new_page()
        page.route("http://studio.test/**", serve)
        page.goto("http://studio.test")
        tip = page.get_by_role("tooltip")
        status = page.locator("#status")
        expect(status).to_contain_text("create a blank model")

        def explains(key):
            expect(tip).to_have_text(catalog[key]["text"])

        # Option can be pressed after hovering and kept down across controls.
        status.hover()
        expect(tip).to_have_count(0)
        page.keyboard.down("Alt")
        explains("studio.status")
        page.locator("#zoom").hover()
        explains("studio.zoom")
        page.locator("#viewport").hover(position={"x": 5, "y": 5})
        expect(tip).to_have_count(0)
        page.keyboard.up("Alt")

        # A native modal has its own top layer. Preserve the input's existing help.
        page.locator("[data-cmd=keys]").click()
        field = page.locator("#keymap-keys")
        original = field.get_attribute("aria-describedby")
        assert original
        field.hover()
        page.keyboard.down("Alt")
        explains("keys.shortcut")
        assert original in field.get_attribute("aria-describedby").split()
        expect(page.locator("dialog[open]").get_by_role("tooltip")).to_be_visible()
        page.keyboard.press("Escape")
        expect(tip).to_have_count(0)
        expect(field).to_be_visible()
        expect(field).to_have_attribute("aria-describedby", original)
        page.keyboard.up("Alt")

        # A disabled command still explains what it does.
        page.locator("dialog[open]").get_by_role("button", name="Cancel").click()
        page.locator("[data-cmd=palette]").click()
        page.locator("#palette-input").fill("Rename selected variable")
        command = page.get_by_role("option").first
        expect(command).to_be_disabled()
        command.hover(force=True, position={"x": 5, "y": 5})
        page.keyboard.down("Alt")
        explains("command.rename")
        page.keyboard.up("Alt")
        expect(tip).to_have_count(0)
    print("Passed Studio Option-hover, disabled command, dialog and accessibility checks.")


if __name__ == "__main__":
    main()
