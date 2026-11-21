"""Exercise negative-control wiring without launching sampler training."""

from collections import Counter

from experiments.hierarchy.discovery import run_phase4_null
from experiments.hierarchy.posterior import PosteriorProblem
from experiments.hierarchy.posterior.exact import enumerate_structures


def test_frozen_null_generators_and_source_provenance():
    candidates = tuple(enumerate_structures(
        PosteriorProblem(n_leaves=4, max_depth=2, infer_controls=True)))
    assert {"discovery/control_stream.py", "discovery/run_phase4_null.py"} <= set(
        run_phase4_null._sources())
    for seed in run_phase4_null.SEEDS:
        records = {arm: run_phase4_null._records(arm, seed, candidates)
                   for arm in run_phase4_null.ARMS}
        assert all(len(rows) == 16 for rows in records.values())
        for arm in ("randomized_null", "exogenous"):
            assert all(count == 2 for count in Counter(
                (row.x, row.action) for row in records[arm]).values())
        observed = records["observational_confounding"]
        shuffled = records["shuffled_observational"]
        assert Counter((row.x, row.y) for row in observed) == Counter(
            (row.x, row.y) for row in shuffled)
        assert Counter((row.x, row.action) for row in observed) == Counter(
            (row.x, row.action) for row in shuffled)


def test_null_cell_exact_update_with_stub_sampler(monkeypatch):
    class StubSampler:
        stats = {"steps": 1000, "trajectories": 64_000,
                 "unique_reward_calls": 1}

        def sample(self, count, *, seed):
            return [self.structure] * count

    candidates = tuple(enumerate_structures(
        PosteriorProblem(n_leaves=4, max_depth=2, infer_controls=True)))
    sampler = StubSampler()
    sampler.structure = candidates[0]
    calls = []

    def train_sampler(problem, history, *, seed, steps, batch_size):
        calls.append((len(history), seed, steps, batch_size))
        return sampler

    monkeypatch.setattr(run_phase4_null, "train_sampler", train_sampler)
    result = run_phase4_null.run_null_cell("randomized_null", 11)
    assert len(result["rounds"]) == len(result["training"]) == 16
    assert [row[0] for row in calls] == list(range(16))
    assert all(row[2:] == (1000, 64) for row in calls)
    assert result["generator_note"] == "no actuator effect"
