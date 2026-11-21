from experiments.hierarchy.discovery.control_acquisition import (
    best_joint_probe,
    joint_information_scores,
    joint_reward_scores,
)


def test_active_probe_can_choose_actuator_to_identify_control():
    # Models are observationally identical under action zero. Their response
    # to action one differs, so structure-only observation cannot settle them.
    def predictive(structure, x, action):
        if action == 0:
            return (0.7, 0.3)
        if structure == "off":
            return (0.7, 0.3)
        return (0.1, 0.9)

    scores = joint_information_scores(["off", "on"], 2, predictive)
    assert scores[0, 0] == 0
    assert scores[1, 0] == 0
    assert scores[0, 1] > 0
    assert best_joint_probe(scores)[1] == 1
    reward = joint_reward_scores(["off", "on"], 2, predictive)
    assert best_joint_probe(reward)[1] == 0
