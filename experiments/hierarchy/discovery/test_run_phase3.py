"""Integration smoke test; never substitutes fake draws in a reported study."""

from experiments.hierarchy.discovery import run_phase3
from experiments.hierarchy.posterior import PosteriorProblem
from experiments.hierarchy.posterior.exact import enumerate_structures


def test_actual_model_schema_and_oracle_update_with_stub_training(monkeypatch):
    candidates = tuple(enumerate_structures(PosteriorProblem(n_leaves=4, max_depth=2)))
    calls = []

    class StubPolicy:
        stats = {"steps": 200, "trajectories": 12800,
                 "unique_reward_calls": 4, "reward_seconds": 0.0}

        def sample(self, n, seed):
            assert n == 256
            return [candidates[i % 4] for i in range(n)]

    def stub_train(problem, history, *, seed, steps, batch_size):
        calls.append((len(history), steps, batch_size))
        return StubPolicy()

    monkeypatch.setattr(run_phase3, "train_sampler", stub_train)
    result = run_phase3.run_cell(0, 11, "information", 200)
    assert len(calls) == 10
    assert calls == [(i, 200, 64) for i in range(10)]
    assert len(result["rounds"]) == 10
    assert all(0 <= row["exact_truth_mass"] <= 1 for row in result["rounds"])
    assert len(result["training"]) == 10
