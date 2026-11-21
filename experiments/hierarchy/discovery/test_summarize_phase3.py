import pytest

from experiments.hierarchy.discovery.summarize_phase3 import summarize


def test_summary_refuses_incomplete_matrix():
    with pytest.raises(ValueError, match="incomplete or extraneous"):
        summarize([], bootstrap_draws=10)
