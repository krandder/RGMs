"""Keep independent replay strict about every non-timing result."""

import pytest

from experiments.hierarchy.discovery.verify_replay import without_timings, verify_replay


def test_only_elapsed_seconds_are_ignored():
    original = {"rounds": [{"exact_truth_mass": 0.2,
                            "train_and_sample_seconds": 12.0}],
                "training": [{"reward_seconds": 0.3, "steps": 1000}]}
    replay = {"rounds": [{"exact_truth_mass": 0.2,
                          "train_and_sample_seconds": 15.0}],
              "training": [{"reward_seconds": 0.5, "steps": 1000}]}
    assert without_timings(original) == without_timings(replay)
    replay["training"][0]["steps"] = 999
    assert without_timings(original) != without_timings(replay)


def test_phase4_replay_requires_release_flag(tmp_path):
    with pytest.raises(ValueError, match="gated"):
        verify_replay("phase4", tmp_path / "reference.json",
                      tmp_path / "rerun.json", tmp_path / "receipt.json")
