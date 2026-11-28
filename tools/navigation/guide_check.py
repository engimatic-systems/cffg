"""Check a rendered navigation guide offline against its accompanying study.json."""

import argparse
import json
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


def check_view(page, root, study, mobile):
    errors = []
    external = []
    page.on("pageerror", lambda exc: errors.append(str(exc)))
    page.on("request", lambda request: external.append(request.url)
            if request.url.startswith(("http:", "https:")) else None)
    page.goto((root / "index.html").as_uri()
              + "?case=aliased&phase=navigation&policy=split&route=amber-to-blue&step=2#replay")
    ui = page.locator
    expect(ui("#error")).to_be_hidden()
    expect(page.get_by_role("heading", name="What should the traveler remember?")).to_be_attached()
    expect(ui(".intro ol li")).to_have_count(3)
    expect(ui("#route")).to_have_value("amber-to-blue")
    expect(ui("#policy")).to_have_value("split")
    expect(ui("#scrub")).to_have_value("2")
    expect(ui("#readings dd").first).to_have_text("junction")
    expect(ui("#readings dd").nth(3)).to_have_text("East")
    ui("#policy").select_option("appearance")
    expect(ui("#route")).to_have_value("amber-to-blue")
    expect(ui("#scrub")).to_have_value("2")
    expect(ui("#readings dd").nth(2)).to_have_text("junction")
    expect(ui("#readings dd").nth(3)).to_have_text("North")
    assert len(study["cases"]) == 4
    expect(ui("#score-table tbody tr")).to_have_count(8)
    expect(ui("#route-table tbody tr")).to_have_count(12)
    if study.get("scale", {}).get("status") == "complete":
        rows = study["scale"]["rows"]
        expect(ui("#scale-table tbody tr")).to_have_count(len(rows))
        for i, r in enumerate(rows):
            warm_ms = 1000 * r["warm_s"]["median"]
            expect(ui("#scale-table tbody tr").nth(i).locator("td").nth(5)).to_have_text(f"{warm_ms:.2f}")
        with page.expect_download() as event:
            ui("#scale-download").click()
        assert json.loads(Path(event.value.path()).read_bytes()) == study["scale"]
    else:
        expect(ui("#scale")).to_be_hidden()
    for i, case in enumerate(study["cases"]):
        finding = next(f for f in study["findings"] if f["id"] == case["id"])
        ui("#fixture").select_option(case["condition"])
        ui("#orientation").select_option(str(case["rotation"]))
        for j, model in enumerate(("appearance", "split")):
            row = ui("#score-table tbody tr").nth(2 * i + j)
            expect(row.locator("td").last).to_have_text(f"{finding['scores'][model]['final']['mean_bfe']:.3f}")
        for j, model in enumerate(("appearance", "split", "tile")):
            stats = finding["routes"][model]
            row = ui("#route-table tbody tr").nth(3 * i + j)
            expect(row.locator("td").nth(1)).to_have_text(f"{stats['completed']} / {stats['attempted']}")
            expect(row.locator("td").last).to_have_text(str(stats["actions"]))
        # The return trip must follow the recorded policy, including failures.
        ui("#phase").select_option("navigation")
        ui("#policy").select_option("selected")
        tape = next(r for r in case["routes"][finding["selection"]["selected"]] if r["id"] == "harbor-to-home")
        ui("#route").select_option(tape["id"])
        ui("#scrub").focus()
        page.keyboard.press("End")
        expect(ui("#step")).to_have_text(f"{len(tape['steps'])} / {len(tape['steps'])}")
        expect(ui("#outcome")).to_contain_text("Goal reached" if tape["success"] else "Move limit reached")
        expect(ui("#readings dd").first).to_have_text(tape["steps"][-1]["next_symbol"])

    ui("#phase").select_option("fit")
    ui("#scrub").focus()
    page.keyboard.press("Home")
    observed = ui("#map rect").count()
    ui("#view").select_option("actual")
    assert observed == 1 and ui("#map rect").count() == len(study["cases"][-1]["layout"]["tiles"])
    ui("#view").select_option("observed")
    ui("#next").click()
    expect(ui("#step")).to_contain_text("1 /")
    ui("#prev").click()
    ui("#play").click()
    expect(ui("#step")).to_contain_text("1 /")
    ui("#play").click()
    expect(ui("#play")).to_have_text("Play")
    ui("#phase").select_option("final")
    ui("#policy").select_option("appearance")
    ui("#next").click()
    case = study["cases"][-1]
    final = next(f for f in study["findings"] if f["id"] == case["id"])["scores"]["appearance"]["final"]
    expect(ui("#readings dd").last).to_have_text(f"{final['energies'][0]:.3f} nats")
    ui("#zoom").click()
    expect(ui("#zoom")).to_have_attribute("aria-pressed", "true")
    ui("#zoom").click()
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), "Page overflow"
    assert not errors, errors
    assert not external, external
    print(f"Passed {'mobile' if mobile else 'desktop'} offline replay, score tables and controls.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Rendered study directory with index.html and study.json")
    parser.add_argument("--chromium", help="Optional path to a local Chromium executable")
    args = parser.parse_args()
    root = args.directory.resolve()
    study = json.loads((root / "study.json").read_bytes())
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=args.chromium)
        try:
            for width in (1280, 390):
                page = browser.new_page(viewport={"width": width, "height": 850},
                                        reduced_motion="reduce", offline=True)
                check_view(page, root, study, width < 600)
                page.close()
        finally:
            browser.close()


if __name__ == "__main__":
    main()
