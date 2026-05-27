"""parse_strategy_output + generate_strategy — port of cells 7, 8.

We keep paper's prompt text verbatim. The only structural change is
that cell 8 references the global `punishment_mechanism`; we read it
from config (which is set once at startup, semantically identical).
"""
from concurrent.futures import ThreadPoolExecutor
import random

import config
from agent import Agent
from llm_qwen import promptLLM
from experiment_log import log_event


def parse_strategy_output(output):
    """Paper cell 7, verbatim."""
    parts = output.split("My strategy will be", 1)
    if len(parts) == 2:
        strategy_justification = parts[0].strip()
        strategy = "My strategy will be" + parts[1].strip()
    else:
        strategy_justification = ""
        strategy = output.strip()
    return strategy_justification, strategy


def generate_strategy(agent_name: str, generationNumber: int,
                      inherited_strategies) -> tuple:
    """Paper cell 8, verbatim prompt text. Returns (justification, strategy)."""
    pm = config.punishment_mechanism
    punishmentLoss = config.punishmentLoss

    if pm == "costly_punishment":
        punishment_text = (
            f"You may also choose to punish the recipient by spending x units "
            f"to take away {punishmentLoss}x of their resources. "
            f"Bear in mind that others may punish you too."
        )
    elif pm == "partner_choice":
        punishment_text = (
            "You may also choose to punish the recipient by refraining from "
            "any interaction. Bear in mind that others may punish you too."
        )
    else:
        punishment_text = ""

    if generationNumber == 1:
        prompt = (
            f"Your name is {agent_name}. "
            "Based on the description of the game, create a strategy that you will follow in the game. "
            "As a donor, you will receive limited information about the recipient and their recent interactions:"
            "You will see what the recipient (let's call them A) did in the immediately preceding round only."
            "You will also see what A's partner from that round (let's call them B) did in their preceding round."
            "This chain continues for at most three rounds back."
            "Important: You do NOT have access to multiple rounds of information about A directly. You only know what A did in the single most recent round."
            "Example (assuming it's now round 4):"
            "You are paired with A."
            "You know: In round 3, A donated X% to B."
            "You also know: In round 2, B donated Y% to C."
            "And finally: In round 1, C donated Z% to D."
            "Remember: This is all the information you have. You don't know what A did in rounds 1 or 2, only in round 3."
            f"{punishment_text}"
            "In the first round, there will be no information about the recipient's previous behavior - think carefully about how much to donate in that case."
            "Before formulating your strategy, briefly think step by step about what would be a successful strategy in this game."
            "Then describe your strategy briefly without explanation in one sentence that starts: My strategy will be."
        )
    else:
        prompt = (
            f"Your name is {agent_name}. "
            "How would you approach the game? Here is the advice of the best-performing 50% of the previous generation, along with their final scores."
            "Modify this advice to create your own strategy."
            f"{inherited_strategies}"
            "As a donor, you will receive limited information about the recipient and their recent interactions:"
            "You will see what the recipient (let's call them A) did in the immediately preceding round only."
            "You will also see what A's partner from that round (let's call them B) did in their preceding round."
            "This chain continues for at most three rounds back."
            "Important: You do NOT have access to multiple rounds of information about A directly. You only know what A did in the single most recent round."
            "Example (assuming it's now round 4):"
            "You are paired with A."
            "You know: In round 3, A donated X% to B."
            "You also know: In round 2, B donated Y% to C."
            "And finally: In round 1, C donated Z% to D."
            "Remember: This is all the information you have. You don't know what A did in rounds 1 or 2, only in round 3."
            f"{punishment_text}"
            "In the first round, there will be no information about the recipient's previous behavior - think carefully about how much to donate in that case."
            "Before formulating your strategy, briefly think step by step about what would be a successful strategy in this game. In particular, think about how you can improve on the surviving agents' strategies."
            "Then describe your strategy briefly without explanation in one sentence that starts: My strategy will be."
        )

    strategy_output = promptLLM(prompt)
    strategy_justification, strategy = parse_strategy_output(strategy_output)

    print(f"{agent_name}:\n  Justification: {strategy_justification[:100]}...\n  Strategy: {strategy[:100]}...")

    log_event(
        type="strategy",
        agent_name=agent_name,
        generation=generationNumber,
        model=config.llm,
        had_inherited=generationNumber > 1,
        prompt=prompt,
        raw_output=strategy_output,
        justification=strategy_justification,
        strategy=strategy,
    )
    return strategy_justification, strategy


def initializeAgents(numAgents: int, initialEndowment: int,
                     generationNumber: int, inherited_strategies) -> list:
    """Paper cell 9, verbatim. Creates `numAgents` agents in parallel via
    ThreadPoolExecutor and shuffles the resulting list.
    """
    agents = []
    with ThreadPoolExecutor() as executor:
        futures = []
        for i in range(numAgents):
            name = f"{generationNumber}_{i + 1}"
            futures.append(executor.submit(
                generate_strategy, str(name), generationNumber, inherited_strategies))

        for i, future in enumerate(futures):
            strategy_justification, new_strategy = future.result()
            name = f"{generationNumber}_{i + 1}"
            agents.append(Agent(
                name=name,
                reputation=False,
                resources=initialEndowment,
                strategy=new_strategy,
                strategy_justification=strategy_justification,
            ))

    random.shuffle(agents)
    return agents
