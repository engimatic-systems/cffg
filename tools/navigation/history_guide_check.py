"""Exercise a rendered movement-history guide offline with its own recordings."""

import argparse
import json
from pathlib import Path
from urllib.parse import urlencode

from playwright.sync_api import expect, sync_playwright


def check(page, root, summary, cases):
    errors, remote = [], []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("request", lambda r: remote.append(r.url) if r.url.startswith(("http:", "https:")) else None)
    url = (root / "index.html").resolve().as_uri()
    page.goto(url)
    ui = page.locator
    expect(ui("#scores tbody tr")).to_have_count(2 * len(cases))
    expect(ui("#routes tbody tr")).to_have_count(3 * len(cases))
    order = ui("#layout option").evaluate_all("(options) => options.map(o => o.value)")
    arms = ("appearance", "history", "tile")
    for i, name in enumerate(order):
        result = summary["layouts"][name]
        for j, lead in enumerate(("1", "3")):
            row = ui("#scores tbody tr").nth(2 * i + j)
            nll = result["scores"][lead]["all"]["mean_nll"]
            for k, arm in enumerate(arms):
                expect(row.locator("td").nth(k + 2)).to_have_text(f"{nll[arm]:.3f}")
        for j, arm in enumerate(arms):
            row = ui("#routes tbody tr").nth(3 * i + j)
            r = result["routes"][arm]
            expect(row.locator("td").nth(1)).to_have_text(f"{r['successes']} / {r['total']}")
            expect(row.locator("td").nth(2)).to_have_text(str(r["actions"]))

    if ui("#example").is_visible():
        ui("#example").click()
        expect(ui("#outcome")).to_contain_text("Goal missed")
        page.go_back()
    if ui("#forecast-example").is_visible():
        ui("#forecast-example").click()
        expect(ui("#horizon")).to_have_value("3")
        expect(ui("#readings")).to_contain_text("Two-move history row")
        page.go_back()

    name = order[-1]
    case = cases[name]
    missed = next(t for t in case["routes"] if not t["success"])
    params = dict(layout=name, phase="route", arm=missed["arm"], tape=missed["id"], step=len(missed["steps"]))
    page.goto(url + "?" + urlencode(params) + "#replay")
    expect(ui("#tape")).to_have_value(missed["id"])
    expect(ui("#outcome")).to_contain_text("Goal missed")
    # Preserve the task and cursor when visiting its same-information map peer.
    peer = next(t for t in case["routes"] if t["id"] == missed["id"] and t["arm"] == "tile")
    ui("#arm").select_option("tile")
    expect(ui("#tape")).to_have_value(missed["id"])
    expect(ui("#scrub")).to_have_value(str(min(len(peer["steps"]), len(missed["steps"]))))
    ui("#view").select_option("fit")
    expect(ui("#caption")).to_contain_text("original learned")
    ui("#zoom").click()
    expect(ui("#zoom")).to_have_attribute("aria-pressed", "true")
    ui("#zoom").click()
    event = next(e for e in case["predictions"] if e["horizon"] == 3 and e["t"] >= 2)
    params.update(phase="forecast", arm="history", tape=event["tape"], step=event["t"], horizon=3)
    page.goto(url + "?" + urlencode(params) + "#replay")
    expect(ui("#past li")).to_have_count(2)
    for arm in arms:
        ui("#arm").select_option(arm)
        expect(ui("#readings")).to_contain_text(f"{event['arms'][arm]['probability'] * 100:.2f}%")
        expect(ui("#readings")).to_contain_text(f"{event['arms'][arm]['bfe']:.3f} nats")
    ui("#scrub").focus()
    page.keyboard.press("End")
    expect(ui("#outcome")).to_contain_text("no complete forecast window")
    page.keyboard.press("Home")
    expect(ui("#past li")).to_have_count(0)
    ui("#next").click()
    expect(ui("#scrub")).to_have_value("1")
    ui("#play").click()
    expect(ui("#scrub")).to_have_value("2")
    ui("#play").click()
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    assert not errors and not remote, (errors, remote)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--chromium", help="Optional local Chromium executable")
    args = parser.parse_args()
    root = args.directory.resolve()
    summary = json.loads((root / "summary.json").read_bytes())
    cases = json.loads((root / "evaluation.json").read_bytes())
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=args.chromium)
        for width in (1280, 390):
            page = browser.new_page(viewport={"width": width, "height": 850}, offline=True)
            check(page, root, summary, cases)
            page.close()
            print(f"Passed at {width}px.")
        browser.close()


if __name__ == "__main__":
    main()
