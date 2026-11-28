"""Run with uv run --extra studio --with playwright python tools/check_studio_tour_motion.py."""

import argparse
from urllib.parse import urlparse

from playwright.sync_api import expect, sync_playwright

from cffg.studio.models import starter
from cffg.studio.tours import edit_tour
from check_studio_help import serve


def model_routes():
    doc = starter("kalman", "Tour motion", steps=12)
    steps = [
        {"title": title, "text": title, "ref": ref}
        for title, ref in (("First transition", "factor:trans1"), ("Last transition", "factor:trans11"))
    ]
    edit_tour(doc, {"action": "set_tour", "id": "travel", "title": "Transitions", "steps": steps})
    model = {**doc.describe(), "id": "fixture", "revision": "fixture-head"}

    def route(request):
        path = urlparse(request.request.url).path
        if path == "/v1/models":
            request.fulfill(json=[{"id": model["id"], "name": doc.template.name, "recipe": "kalman"}])
        elif path == "/v1/models/fixture":
            request.fulfill(json=model)
        else:
            serve(request)

    return route


def travel(page, key, duration=1700):
    page.evaluate("""() => {
        const layer = document.querySelector('.graph-canvas > g');
        function sample() {
            const {a,e,f} = layer.getCTM();
            return [a,e,f];
        }
        window.poses = [sample()];
        window.cameraObserver?.disconnect();
        window.cameraObserver = new MutationObserver(() => window.poses.push(sample()));
        window.cameraObserver.observe(layer, {attributes:true, attributeFilter:['transform']});
    }""")
    page.keyboard.press(key)
    page.clock.run_for(duration)
    return page.evaluate("window.poses")


def target(page):
    return page.locator(".tour-frame").evaluate("""el => {
        const {x,y,width,height} = el.getBBox();
        return [x,y,x+width,y+height];
    }""")


def check_flight(page, key):
    departure = target(page)
    poses = travel(page, key)
    arrival = target(page)
    wide = min(scale for scale, x, y in poses)
    assert wide < .8 * min(poses[0][0], poses[-1][0]), "Expected a pullback and a zoom into the target"
    pan = [pose for pose in poses if pose[0] == wide]
    assert len(pan) > 3 and pan[0][1:] != pan[-1][1:], "Expected movement at the wide scale"
    scale, tx, ty = pan[len(pan) // 2]
    svg = page.locator(".graph-canvas").bounding_box()
    for x1, y1, x2, y2 in (departure, arrival):
        assert 0 <= x1*scale+tx <= x2*scale+tx <= svg["width"]
        assert 0 <= y1*scale+ty <= y2*scale+ty <= svg["height"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--browser", choices=["chromium", "webkit"], default="chromium")
    args = parser.parse_args()
    with sync_playwright() as pw, getattr(pw, args.browser).launch() as browser:
        page = browser.new_page(viewport={"width": 1440, "height": 900}, reduced_motion="no-preference")
        page.route("http://studio.test/**", model_routes())
        page.clock.install()
        page.goto("http://studio.test/?model=fixture&tour=travel")
        toggle = page.get_by_role("checkbox", name="Animations")
        expect(toggle).to_be_checked()
        expect(page.locator(".tour-frame")).to_be_visible()
        page.clock.run_for(1700)
        check_flight(page, "ArrowRight")
        check_flight(page, "ArrowLeft")

        travel(page, "ArrowRight", duration=180)
        toggle.focus()
        page.keyboard.press("Space")
        expect(toggle).not_to_be_checked()
        page.clock.run_for(40)
        landing = page.locator(".graph-canvas > g").first.get_attribute("transform")
        page.clock.run_for(1700)
        assert page.locator(".graph-canvas > g").first.get_attribute("transform") == landing
        assert len(travel(page, "ArrowLeft")) <= 2
        page.reload()
        expect(toggle).not_to_be_checked()
        page.emulate_media(reduced_motion="reduce")
        toggle.check()
        page.clock.run_for(30)
        check_flight(page, "ArrowRight")
        page.evaluate("localStorage.clear()")
        page.reload()
        expect(toggle).not_to_be_checked()
        page.goto("http://studio.test/?model=fixture&tour=intro")
        toggle.check()
        ring = page.locator(".tour-ring")
        page.keyboard.press("ArrowRight")
        assert ring.evaluate("el => el.getAnimations().length") == 1
        page.clock.run_for(900)
        toggle.uncheck()
        page.keyboard.press("ArrowLeft")
        assert ring.evaluate("el => el.getAnimations().length") == 0
        print(f"Passed {args.browser} model and intro tour routes, toggle, persistence, and reduced motion.")


if __name__ == "__main__":
    main()
