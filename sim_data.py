"""SimulationData + AgentRoundData — verbatim port of notebook cell 6."""
from dataclasses import dataclass, field


@dataclass
class SimulationData:
    hyperparameters: dict
    agents_data: list = field(default_factory=list)

    def to_dict(self):
        return {
            "hyperparameters": self.hyperparameters,
            "agents_data": self.agents_data,
        }


@dataclass
class AgentRoundData:
    agent_name: str
    round_number: int
    game_number: int
    paired_with: str
    current_generation: int
    resources: int
    donated: float
    received: float
    strategy: str
    strategy_justification: str
    reputation: float
    is_donor: bool
    traces: list
    history: list
    justification: str = ""
    punished: bool = False
