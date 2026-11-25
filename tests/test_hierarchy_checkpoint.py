"""Restore the exact inter-level boundary and native planner's staged goal messages."""

from dataclasses import replace
import zipfile

import numpy as np
import pytest

from cffg.hierarchy import Hierarchy
from cffg.hierarchy_model import canonical_model
from test_hierarchy import tables


def build(*, thickness=3, goals=None):
    return Hierarchy.from_tables(*tables(), thickness=thickness, goal_window=goals)


def test_checkpoint_resume_across_both_clocks_and_reject_incompatible_state(tmp_path):
    run = build()
    run.step(1).require_converged()
    snap = run.checkpoint()
    restored = build()
    restored.restore(snap)
    snap["cost"].fill(9000)
    np.testing.assert_array_equal(restored.checkpoint()["cost"],run.checkpoint()["cost"])
    saved = tmp_path / "coupled.cffgz"
    run.save_bundle(saved)
    bundled = Hierarchy.load_bundle(saved)
    assert bundled.spec == run.spec and bundled.artifact_hash == run.artifact_hash
    for observation in (0,None,1,0):
        outputs = [h.step(observation).require_converged() for h in (run,restored,bundled)]
        for other in outputs[1:]:
            np.testing.assert_array_equal(other.joint,outputs[0].joint)
            np.testing.assert_array_equal(other.log_cavity,outputs[0].log_cavity)
            assert other.ledger == outputs[0].ledger
    snap = run.checkpoint()
    for name, bad in (("prior",np.array([np.nan,0.])), ("periods",np.array(182)), ("version",np.array(0)),
                      ("frontier",np.zeros((1,1))), ("cost",snap["cost"] + 1), ("child_statistics",np.zeros_like(snap["child_statistics"]))):
        with pytest.raises(ValueError,match="checkpoint"):
            restored.restore({**snap,name:bad})
        for k,v in restored.checkpoint().items():
            np.testing.assert_array_equal(v,snap[k])
    with pytest.raises(ValueError,match="unversioned"):
        restored.restore({})
    with pytest.raises(ValueError,match="different artifact"):
        build(thickness=4).restore(snap)
    p,t,d,b,a = tables()
    with pytest.raises(ValueError,match="different artifact"):
        Hierarchy.from_tables(p[::-1],t,d,b,a,thickness=3).restore(snap)
    with zipfile.ZipFile(saved) as original, zipfile.ZipFile(tmp_path/"changed", "w") as altered:
        for name in original.namelist():
            raw = original.read(name)
            altered.writestr(name, raw+b"!" if name=="checkpoint.npz" else raw)
    with pytest.raises(ValueError,match="digest mismatch"):
        Hierarchy.load_bundle(tmp_path/"changed")


def test_scoped_names_and_both_templates_are_serializable(tmp_path):
    from cffg.core.compose import scoped
    spec, parent, pa, child, ca = canonical_model(*tables(),1)
    parent, child = scoped(parent,"slow layer"), scoped(child,"fast layer")
    pa = {"slow layer/"+k: v for k,v in pa.items()}
    ca = {"fast layer/"+k: v for k,v in ca.items()}
    spec = replace(spec,parent_template=parent.name,child_template=child.name,
                   parent_port=next(iter(parent.ports)),child_port=next(k for k,p in child.ports.items()
                       if p.role.value=="glue"),child_transition=next(k for k,f in child.factors.items()
                                                                  if f.type=="transition_mixture"))
    h = Hierarchy(spec,parent,pa,child,ca)
    file = tmp_path/"named.cffgz"
    h.save_bundle(file)
    restored = Hierarchy.load_bundle(file)
    assert restored.spec == spec
    np.testing.assert_array_equal(restored.step(1).joint,build(thickness=1).step(1).joint)


def test_learned_goal_matches_standalone_through_replanning_pending_restore_and_rollover(tmp_path):
    from cffg.compiler import compile_template
    from cffg.engine import Engine
    from test_goal_hyperprior import _model, _arrays
    template, arrays = _model(), _arrays()
    run = build(goals=(template,arrays))
    standalone = Engine(compile_template(template,arrays=arrays))
    previous = {"Cc":arrays["Cc"]}
    for episode in range(2):
        state = standalone.with_goal_priors(standalone.init_state(),previous)
        ref = standalone.run(state,100,tol=1e-10,harden_last=0).require_converged()
        expected = standalone.goal_prior_updates(ref)
        for _ in range(2):
            run.plan_goals(100,tol=1e-10).require_converged()
            run.preview(None).require_converged()
        np.testing.assert_array_equal(run.goal_priors["Cc"],previous["Cc"])
        if episode == 0:
            run.step(0).require_converged()
            path = tmp_path/"goal-hierarchy.cffgz"
            run.save_bundle(path)
            run = Hierarchy.load_bundle(path)
            assert run.child_ticks == 1 and run.parent_periods == 0
        assert run.finish_episode()
        np.testing.assert_array_equal(run.goal_priors["Cc"],expected["Cc"])
        assert not run.finish_episode()
        previous = expected
