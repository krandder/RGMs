"""Validate the frozen phase-4 truth selection without running a study."""

from experiments.hierarchy.discovery.run_phase4 import _truth_cases
from experiments.hierarchy.posterior import PosteriorProblem
from experiments.hierarchy.posterior.exact import enumerate_structures


def test_truth_cases_cover_no_global_and_local_control():
    candidates = tuple(enumerate_structures(
        PosteriorProblem(n_leaves=4, max_depth=2, infer_controls=True)))
    cases = _truth_cases(candidates)
    assert len(candidates) == 26
    assert cases["none"].controlled == ()
    assert cases["root"].controlled == ((0, 1, 2, 3),)
    assert cases["local"].depth == 2
    assert len(cases["local"].controlled) == 1
    assert cases["local"].controlled[0] != (0, 1, 2, 3)
