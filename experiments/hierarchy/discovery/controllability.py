"""Joint-structure posterior readouts for action-conditioned control flags."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math


def flag_marginals(
    posterior: Mapping[str, float],
    flags_by_structure: Mapping[str, Mapping[str, bool]],
) -> dict[str, dict[str, float | None]]:
    """Marginalize flags by stable node ID across candidate hierarchies.

    Missing nodes and present nodes with the flag off are distinct. The
    unconditional controlled probability includes structure uncertainty;
    conditional controlled probability is reported only when node presence
    has positive posterior mass.
    """
    if not posterior or set(posterior) != set(flags_by_structure):
        raise ValueError("posterior and flag structures must have identical nonempty support")
    if any(not math.isfinite(p) or p < 0 for p in posterior.values()):
        raise ValueError("invalid posterior mass")
    if not math.isclose(math.fsum(posterior.values()), 1.0, rel_tol=1e-9, abs_tol=1e-10):
        raise ValueError("posterior mass must sum to one")
    nodes = sorted({node for flags in flags_by_structure.values() for node in flags})
    if not nodes:
        raise ValueError("at least one candidate node is required")
    output = {}
    for node in nodes:
        present = math.fsum(posterior[key] for key, flags in flags_by_structure.items() if node in flags)
        controlled = math.fsum(posterior[key] for key, flags in flags_by_structure.items()
                               if flags.get(node, False))
        output[node] = {
            "present": present,
            "controlled": controlled,
            "controlled_given_present": controlled / present if present > 0 else None,
        }
    return output


def flag_metrics(
    inclusion: Sequence[float],
    truth: Sequence[bool],
    *,
    threshold: float = 0.5,
) -> dict[str, float | int | None]:
    """Evaluate calibrated inclusion and thresholded flag recovery."""
    if not inclusion or len(inclusion) != len(truth):
        raise ValueError("inclusion and truth must have matching nonempty lengths")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold outside [0,1]")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in inclusion):
        raise ValueError("inclusion probabilities outside [0,1]")
    predicted = [p >= threshold for p in inclusion]
    tp = sum(p and t for p, t in zip(predicted, truth))
    fp = sum(p and not t for p, t in zip(predicted, truth))
    fn = sum(not p and t for p, t in zip(predicted, truth))
    return {
        "threshold": threshold,
        "brier": math.fsum((p - float(t)) ** 2 for p, t in zip(inclusion, truth)) / len(truth),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
    }
