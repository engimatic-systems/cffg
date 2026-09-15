"""simglucose TIR benchmark (side project).

The contract: an unannounced-meal controller (CGM only) must keep the adult
cohort's blood glucose in [70, 180] mg/dL at least 70% of the time — the
FDA-cited lower bound. Reference points on this exact scenario: announced
basal-bolus ~96% (privileged ceiling), basal-only ~34% (floor), a prior
agent's pymdp attempt 74.7%, current reactive controller ~87.6%.

THE REAL BAR (do not chase until CFFG is done — see ticket cff-t2g3): Cooper's
own VERSES-era experiments reached 67% TIR on unannounced adults AND
adolescents with ZERO critical episodes, on 8-virtual-hour episodes. The
end goal is an insulin pump that does continual learning with a white-box
algorithm and a modular physiological world model. Note the reactive
controller here is NOT comparable: adults-only, 24 h, and no critical-episode
accounting — a harder-cohort benchmark variant is part of the later stage.
"""

import numpy as np
import pytest

from cffg.benchmarks.glucose import ADULTS, run_cohort, run_patient

FDA_TIR_BOUND = 0.70


@pytest.mark.slow
def test_reactive_controller_clears_fda_bound():
    """Full 10-adult cohort, 24 h, unannounced meals (~35 s)."""
    result = run_cohort("reactive")
    assert result.tir >= FDA_TIR_BOUND, f"cohort below FDA bound:\n{result.summary()}"
    # No individual patient may be sacrificed for the average, and
    # hypoglycemia must stay rare — TIR bought with lows doesn't count.
    for p in result.patients:
        assert p.tir >= 0.50, f"{p.name} collapsed: {p.tir:.1%}"
        assert p.time_low <= 0.10, f"{p.name} hypo time too high: {p.time_low:.1%}"


def test_benchmark_is_nontrivial():
    """Basal-only floor: if this ever passes 70%, the scenario got too easy
    and the benchmark stopped meaning anything."""
    floor = run_patient("adult#001", "basal_only")
    assert floor.tir < 0.50, f"benchmark too easy: basal-only TIR {floor.tir:.1%}"


def test_benchmark_is_deterministic():
    a = run_patient("adult#002", "reactive")
    b = run_patient("adult#002", "reactive")
    np.testing.assert_array_equal(a.bg, b.bg)


def test_cohort_definition_stable():
    assert ADULTS == [f"adult#{i:03d}" for i in range(1, 11)]
