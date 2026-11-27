"""Graph tour documents. Stable anchors identify stops across renames and revisions."""

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TourStep(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    title: str = Field(min_length=1, max_length=120)
    text: str = Field(min_length=1, max_length=4000)
    ref: str = Field(default="", max_length=512)
    plate: str = Field(default="", max_length=256)


class Tour(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")
    title: str = Field(min_length=1, max_length=120)
    steps: list[TourStep] = Field(min_length=1, max_length=128)

    @field_validator("id")
    @classmethod
    def reserve_intro(cls, value):
        if value == "intro":
            raise ValueError("The intro tour belongs to the workstation")
        return value


def edit_tour(doc, command):
    action = command["action"]
    tours = doc.meta.setdefault("tours", {})
    if action == "remove_tour":
        key = command.get("id")
        if not isinstance(key, str) or key not in tours:
            raise ValueError("Unknown tour ID")
        del tours[key]
        return
    tour = Tour.model_validate({k: v for k, v in command.items() if k != "action"})
    if tour.id not in tours and len(tours) >= 100:
        raise ValueError("A model can contain up to 100 tours")
    doc.sync_anchors()
    stops = []
    for i, step in enumerate(tour.steps, 1):
        target = doc
        stop = step.model_dump(exclude_defaults=True)
        if step.plate:
            if step.plate not in doc.template.plates:
                raise ValueError(f"Tour stop {i}: unknown plate {step.plate!r}")
            stop["plate_anchor"] = doc.meta["anchors"]["plate:" + step.plate]
            target = doc.fragment(step.plate)
        if step.ref:
            anchor = target.meta["anchors"].get(step.ref)
            if not anchor:
                raise ValueError(f"Tour stop {i}: unknown graph target {step.ref!r}")
            stop["anchor"] = anchor
        stops.append(stop)
    tours[tour.id] = {"id": tour.id, "title": tour.title, "steps": stops}


def check_portable_tours(metadata):
    """Validate imported prose and target IDs while allowing historical, removed anchors."""
    tours = metadata.get("tours", {})
    if not isinstance(tours, dict) or len(tours) > 100:
        raise ValueError("Invalid Studio tour collection")
    for key, tour in tours.items():
        if (
            not isinstance(tour, dict)
            or key != tour.get("id")
            or not isinstance(tour.get("steps"), list)
        ):
            raise ValueError("Invalid Studio tour")
        stops = []
        for step in tour["steps"]:
            if not isinstance(step, dict):
                raise ValueError("Invalid Studio tour stop")
            if bool(step.get("ref")) != bool(step.get("anchor")) or bool(
                step.get("plate")
            ) != bool(step.get("plate_anchor")):
                raise ValueError(
                    "A targeted tour stop must carry its stable identifier"
                )
            public = dict(step)
            for field in ("anchor", "plate_anchor"):
                if field in public and not re.fullmatch(
                    "[a-f0-9]{32}", str(public.pop(field))
                ):
                    raise ValueError("Invalid Studio tour anchor")
            stops.append(public)
        Tour.model_validate({**tour, "steps": stops})
