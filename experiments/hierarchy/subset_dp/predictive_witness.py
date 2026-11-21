"""Post hoc single-case prediction check; no new sampler-performance benchmark.

Seed 2 of sixteen_depth2 was selected because its archived 1,000-update sampler
had the largest held-out score gain over the exact posterior among the three
depth-two cases. Rebuild that one realized 256-draw mixture and evaluate its
expected predictive loss using the simulator's known kernel, with parameters
and structure weights fixed after training. This does not estimate general
performance over training seeds, draws or worlds.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import numpy as np
import scipy
from experiments.hierarchy.posterior import model
from experiments.hierarchy.posterior.model import PosteriorProblem, TemporalWorld, random_structure
from experiments.hierarchy.posterior.tb import train_sampler


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def history_sha(history):
    data = np.asarray([(r.x, r.action, r.y) for r in history], dtype='<i2')
    return hashlib.sha256(data.tobytes()).hexdigest()


def run(scaled_path, exact_path):
    started = time.perf_counter()
    scaled = json.loads(scaled_path.read_text())
    exact = json.loads(exact_path.read_text())
    config = scaled['config']
    versions = {'python': platform.python_version(), 'numpy': np.__version__,
                'scipy': scipy.__version__}
    if any(versions[key] != config[key] for key in versions):
        raise RuntimeError(f'Use the archived environment: { {k: config[k] for k in versions} }; got {versions}')
    thread_vars = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')
    if any(os.environ.get(key) != '1' for key in thread_vars):
        raise RuntimeError('Set OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1')
    posterior_dir = Path(model.__file__).resolve().parent
    core_files = ('__init__.py', 'model.py', 'grammar.py', 'tb.py', 'baselines.py', 'run_benchmark.py')
    core_hashes = {name: sha(posterior_dir / name) for name in core_files}
    assert all(value == config['source_sha256'][name] for name, value in core_hashes.items())
    case, seed, budget, draw_count = 'sixteen_depth2', 2, 1000, 256
    archived = next(row for row in scaled['rows'] if
                    (row['case'], row['seed'], row['method'], row['budget']) ==
                    (case, seed, 'gflownet', budget))
    reference = next(row for row in exact['rows'] if
                     (row['case'], row['seed']) == (case, seed))
    problem = PosteriorProblem(16)
    truth = random_structure(16, 2, seed=seed + 19)
    world = TemporalWorld(truth, seed=seed + 31, iid=False)
    history = world.sample(512, seed=seed + 101, randomized_actions=False)
    heldout = world.sample(512, seed=seed + 1001, randomized_actions=False)
    assert json.loads(json.dumps(truth.to_dict())) == archived['truth'] == reference['truth']
    assert history_sha(history) == reference['hist_sha256']
    log_weight = problem.log_weight(truth, history)
    assert abs(log_weight - archived['truth_log_weight']) < 1e-10
    assert abs(log_weight - reference['truth_log_weight']) < 1e-10
    kernel = np.array([world.predictive(x, 0) for x in range(16)])
    oracle_score = float(np.mean([np.log(kernel[r.x, r.y]) for r in heldout]))
    assert abs(oracle_score - archived['oracle_mean_log_predictive']) < 1e-12
    print('Archive, environment, source, truth and training-history identities verified.', flush=True)

    policy = train_sampler(problem, history, seed=seed + 201, steps=budget, batch_size=64)
    draws = policy.sample(draw_count, seed=seed + 301)
    non_timing = lambda stats: {key: value for key, value in stats.items()
                                if key not in ('seconds', 'reward_seconds')}
    assert non_timing(policy.stats) == non_timing(archived['training'])
    depth_counts = Counter(structure.depth for structure in draws)
    assert {str(d): depth_counts[d] / draw_count for d in range(1, 5)} == archived['sample_depth_probabilities']
    truth_frequency = sum(structure.id == truth.id for structure in draws) / draw_count
    assert truth_frequency == archived['sample_truth_frequency']
    mixture = np.zeros((16, 16))
    for structure in draws:
        counts = problem.counts(structure, history)
        mixture += np.array([problem.predictive_from_counts(structure, x, 0, counts)
                             for x in range(16)]) / draw_count
    reference_prediction = np.asarray(reference['exact_predictive_matrix'])
    assert reference_prediction.shape == (16, 16)
    for matrix in (kernel, mixture, reference_prediction):
        assert np.all(np.isfinite(matrix)) and np.all(matrix > 0)
        np.testing.assert_allclose(matrix.sum(axis=1), 1, atol=1e-12, rtol=0)

    occupancy = np.zeros(16)
    source = np.eye(16)[0]
    for _ in range(512):
        occupancy += source / 512
        source = source @ kernel
    # Independent repeated-squaring check of sum_{t=0}^{511} K^t.
    power, total = kernel.copy(), np.eye(16)
    for _ in range(9):
        total = total + power @ total
        power = power @ power
    occupancy_error = float(np.max(np.abs(occupancy - total[0] / 512)))
    assert occupancy_error < 1e-12 and abs(occupancy.sum() - 1) < 1e-12
    scores = {}
    for label, prediction, archived_score in (
        ('trained_draw_mixture', mixture, archived['mean_log_predictive']),
        ('exact_posterior', reference_prediction, reference['heldout_mean_log_predictive']),
    ):
        score = float(np.mean([np.log(prediction[r.x, r.y]) for r in heldout]))
        assert abs(score - archived_score) < 1e-12
        scores[label] = {
            'heldout_mean_log_predictive': score,
            'archived_heldout_mean_log_predictive': archived_score,
            'heldout_absolute_reproduction_error': abs(score - archived_score),
            'expected_excess_log_loss_nats': float(np.sum(
                occupancy[:, None] * kernel * np.log(kernel / prediction))),
        }
    return {
        'role': __doc__.strip(),
        'finished_utc': datetime.now(timezone.utc).isoformat(),
        'case': case, 'seed': seed, 'updates': budget, 'draw_count': draw_count,
        'environment': versions,
        'threads': {key: os.environ[key] for key in thread_vars},
        'artifact_sha256': {'scaled.json': sha(scaled_path), 'subset_dp/results.json': sha(exact_path)},
        'check_script_sha256': sha(__file__), 'matched_archived_core_sha256': core_hashes,
        'truth': truth.to_dict(), 'truth_log_weight': log_weight,
        'training_history_sha256': history_sha(history),
        'heldout_history_sha256': history_sha(heldout),
        'training_non_timing_fields_match': True, 'training': policy.stats,
        'sample_depth_counts': dict(sorted(depth_counts.items())),
        'sample_truth_frequency': truth_frequency,
        'unique_sampled_structures': len(set(structure.id for structure in draws)),
        'draws': [structure.to_dict() for structure in draws],
        'evaluation': ('Expected one-step excess log loss under the known generator; '
                       'mean source-state occupancy over 512 transitions starting at leaf zero. '
                       'Models remain fixed after training. No iid standard error is used.'),
        'scores': scores,
        'expected_excess_trained_minus_exact': (
            scores['trained_draw_mixture']['expected_excess_log_loss_nats']
            - scores['exact_posterior']['expected_excess_log_loss_nats']),
        'occupancy_crosscheck_max_error': occupancy_error,
        'occupancy': occupancy.tolist(), 'generator_kernel': kernel.tolist(),
        'trained_draw_predictive_matrix': mixture.tolist(),
        'exact_predictive_matrix': reference_prediction.tolist(),
        'wall_seconds': time.perf_counter() - started,
    }


if __name__ == '__main__':
    posterior = Path(model.__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scaled', type=Path, default=posterior / 'results/scaled.json')
    parser.add_argument('--exact', type=Path, default=posterior.parent / 'subset_dp/results.json')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f'Refusing to overwrite {args.out}')
    result = run(args.scaled, args.exact)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result['scores']), flush=True)
