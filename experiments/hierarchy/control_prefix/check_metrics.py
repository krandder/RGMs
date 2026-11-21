"""Recalculate reported prefix metrics from node marginals, independently of the study code."""
from argparse import ArgumentParser
from hashlib import sha256
import json
from math import fsum, isfinite
from pathlib import Path


def node_sets(tree):
    if isinstance(tree, int):
        return frozenset([tree]), set()
    children = [node_sets(child) for child in tree]
    leaves = frozenset().union(*(child[0] for child in children))
    return leaves, {leaves}.union(*(child[1] for child in children))


def check(path):
    data = json.loads(path.read_text())
    max_error, readouts = 0.0, 0

    def compare(actual, saved):
        nonlocal max_error
        assert isfinite(actual) and isfinite(saved)
        max_error = max(max_error, abs(actual - saved))

    seen = set()
    for cell in data["cells"]:
        case = cell["truth_case"], cell["environment_seed"]
        assert case not in seen
        seen.add(case)
        truth = json.loads(cell["truth_id"])
        _, nodes = node_sets(truth["tree"])
        flags = set(map(frozenset, truth["controlled"]))
        assert flags <= nodes
        assert set(cell["prefixes"]) == {"16", "64", "256"}
        for prefix in cell["prefixes"].values():
            readouts += 1
            m = {frozenset(map(int, k.split(","))): v for k, v in prefix["flag_marginals"].items()}
            assert len(m) == 7 and nodes <= m.keys()
            def include(g):
                return m[g]["controlled"]
            for g, row in m.items():
                assert 0 <= include(g) <= row["present"] + 1e-12
                compare(include(g), row["present"] * row["controlled_given_present"])
            def brier(groups):
                return fsum((include(g) - float(g in flags)) ** 2 for g in groups) / len(groups)
            compare(brier(m), prefix["canonical_node_flag_brier"])
            compare(brier(nodes), prefix["true_tree_node_flag_brier"])
            if flags:
                compare(fsum(include(g) for g in flags) / len(flags), prefix["true_controlled_node_mean_inclusion"])
            else:
                assert prefix["true_controlled_node_mean_inclusion"] is None
            false = {",".join(map(str, sorted(g))): include(g) for g in m.keys() - flags}
            assert false.keys() == prefix["false_control_marginals"].keys()
            for key, value in false.items():
                compare(value, prefix["false_control_marginals"][key])
            compare(max(false.values()), prefix["maximum_false_control_probability"])
    assert seen == {(truth, seed) for truth in ("none", "root", "local") for seed in (11, 23, 37)}
    assert readouts == 27 and max_error < 1e-12, (readouts, max_error)
    return dict(result_sha256=sha256(path.read_bytes()).hexdigest(), prefix_readouts_checked=readouts,
                maximum_error=max_error, passed=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.results), indent=2))
