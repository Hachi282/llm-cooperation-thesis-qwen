"""selectTopAgents + selectRandomAgents + selectHighestReputation —
verbatim port of cell 19."""
import random


def selectTopAgents(agents: list) -> list:
    """Select the top half of agents based on total_final_score."""
    return sorted(agents, key=lambda x: x.total_final_score,
                  reverse=True)[:len(agents) // 2]


def selectRandomAgents(agents: list) -> list:
    """Select half of the agents randomly."""
    return random.sample(agents, len(agents) // 2)


def selectHighestReputation(agents: list) -> list:
    """Select the top half by average_reputation."""
    return sorted(agents, key=lambda agent: agent.average_reputation,
                  reverse=True)[:len(agents) // 2]
