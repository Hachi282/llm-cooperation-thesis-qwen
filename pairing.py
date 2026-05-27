"""bipartiteRoundRobin + extendRounds — verbatim port of cells 10, 11."""


def bipartiteRoundRobin(agents):
    """Paper cell 10. Half the agents are static (group A), the other half
    rotate (group B). Roles toggle between rounds so each pair gets to be
    donor / recipient equally."""
    num_agents = len(agents)
    assert num_agents % 2 == 0, "Number of agents must be even."
    group_A = agents[:num_agents // 2]
    group_B = agents[num_agents // 2:]
    rounds = []
    toggle_roles = False
    for i in range(len(group_A)):
        rotated_group_B = group_B[-i:] + group_B[:-i]
        if toggle_roles:
            round_pairings = list(zip(rotated_group_B, group_A))
        else:
            round_pairings = list(zip(group_A, rotated_group_B))
        rounds.append(round_pairings)
        toggle_roles = not toggle_roles
    return rounds


def extendRounds(original_rounds):
    """Paper cell 11. Append the same rounds with roles swapped, so each
    pair plays both directions within one game."""
    extended_rounds = original_rounds.copy()
    for round in original_rounds:
        reversed_round = [(b, a) for a, b in round]
        extended_rounds.append(reversed_round)
    return extended_rounds
