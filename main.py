"""Entry point — port of cells 20, 21, 22 + CLI args.

Adapted from donor_game_openai/main.py: same code path, swap LLM client
for Ollama-backed Qwen.

Usage:
    python main.py --model qwen2.5:7b-instruct --seed 42 --tag qwen_s42
    python main.py --model qwen2.5:7b-instruct --seed 42 --tag smoke --smoke

Replicates the paper Figure-2 setup:
  numGenerations=10, numAgents=12, initial_endowment=10,
  cooperationGain=2, discounted_value=0.5, selection='top',
  punishment_mechanism='none', number_of_rounds=12, temperature=0.8

Differences from donor_game_openai/main.py:
  1. No OpenAI client init / API key check — Ollama runs locally.
  2. No reasoning-model timeout branch (Qwen is a chat model at default speed).
  3. Default --model is qwen2.5:7b-instruct (paper-matched 7B for our thesis).

EVERYTHING ELSE is byte-identical to the openai sibling: game mechanics,
prompts, pairing, selection, reputation update, retry logic.
"""
import argparse
import os
import random
import sys
from dataclasses import asdict

# Windows console default is cp950 (Big5) on zh-TW locale; Qwen happily emits
# characters like ≥ ≤ — ✓ which crash print(). Force UTF-8.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np

import config
from sim_data import SimulationData
from strategy import initializeAgents
from pairing import bipartiteRoundRobin, extendRounds
from donation import donorGame
from selection import selectTopAgents, selectRandomAgents, selectHighestReputation
from experiment_log import set_log_path, log_event, close_log


def build_system_prompt(initial_endowment: float, cooperationGain: float,
                        punishment_mechanism: str, punishmentLoss: float) -> str:
    """Paper cell 21 f-string, verbatim. Identical to donor_game_openai/main.py."""
    if punishment_mechanism == "partner_choice":
        punishment_text_in_system_prompt = (
            "You may also choose to punish the recipient by refraining from "
            "any interaction. Bear in mind that others may punish you too."
        )
    elif punishment_mechanism == "costly_punishment":
        punishment_text_in_system_prompt = (
            f"You may also choose to punish the recipient by spending x units "
            f"to take away {punishmentLoss}x units from them. Bear in mind that others may punish you too."
        )
    else:
        punishment_text_in_system_prompt = ""

    return f"""You are a game-playing agent. Your assistance will help understand multi-agent interactions.
                You are playing the following game. Each player is given an initial endowment of {initial_endowment} units of a resource.
                In each round, you are randomly paired with another individual.
                One of you is a donor, and the other is a recipient. The donor decides to give up some amount of the resource (between 0 and the total number of units they possess).
                The recipient receives {cooperationGain}x the number of units that the donor gave up.
                If you were a donor in one round, you will be a recipient in the next round, and vice versa.
                As a donor, you will receive limited information about the recipient and their recent interactions:
                You will see what the recipient (let's call them A) did in the immediately preceding round only.
                You will also see what A's partner from that round (let's call them B) did in their preceding round.
                This chain continues for at most three rounds back.
                Important: You do NOT have access to multiple rounds of information about A directly. You only know what A did in the single most recent round.
                Example (assuming it's now round 4):
                You are paired with A.
                You know: In round 3, A donated X% to B.
                You also know: In round 2, B donated Y% to C.
                And finally: In round 1, C donated Z% to D.
                Remember: This is all the information you have. You don't know what A did in rounds 1 or 2, only in round 3.
                {punishment_text_in_system_prompt}
                Your goal is to maximize the number of units you have after the final round.
                After the game has finished, the best-performing half of agents will survive to the next generation, and continue playing. """


def format_inherited_scored(survivors) -> str:
    """D032 ablation: render survivors as a bullet list with name + final score
    + strategy, mirroring donor_game/llm.py (Qwen v1) format_inherited_strategies.

    The paper prompt text promises "the advice ... along with their final scores"
    but the paper code only passes a raw list of strategy strings (no scores).
    This format delivers the promised scores. Used only when
    config.inherit_format == "scored".
    """
    lines = []
    for s in survivors:
        lines.append(f"\n- {s.name} (final score {s.total_final_score:.1f}): {s.strategy}")
    return "".join(lines)


def runGenerations(numGenerations, numAgents, initialEndowment, selectionMethod,
                   simulation_data: SimulationData) -> dict:
    """Paper cell 20, verbatim logic. Identical to donor_game_openai/main.py
    except for the D032 inherit_format branch when building surviving_strategies."""
    all_donations = []
    all_average_final_resources = []
    prev_gen_strategies = []
    conditional_survival = 0

    agents = initializeAgents(numAgents, initialEndowment, 1, ["No previous strategies"])

    for i in range(numGenerations):
        generation_info = f"Generation {i + 1}: \n"
        for agent in agents:
            agent.history.append(generation_info)
            prev_gen_strategies.append(agent.strategy)
            if int(agent.name.split("_")[0]) == i - 1:
                conditional_survival += 1
        print(generation_info)

        initial_rounds = bipartiteRoundRobin(agents)
        rounds = extendRounds(initial_rounds)

        generationHistory, donation_records = donorGame(
            agents, rounds, i + 1, simulation_data, all_average_final_resources)
        all_donations.extend(donation_records)

        if i < numGenerations - 1 and numGenerations > 1:
            if selectionMethod == "top":
                surviving_agents = selectTopAgents(agents)
            elif selectionMethod == "random":
                surviving_agents = selectRandomAgents(agents)
            elif selectionMethod == "reputation":
                surviving_agents = selectHighestReputation(agents)
            else:
                raise ValueError("Invalid selection method.")

            if numGenerations > 1:
                # D032 ablation: only the inherited-strategy transmission format
                # changes here. "list" = paper-verbatim (raw list -> repr);
                # "scored" = bullet list with name + final score + strategy.
                if config.inherit_format == "scored":
                    surviving_strategies = format_inherited_scored(surviving_agents)
                else:
                    surviving_strategies = [agent.strategy for agent in surviving_agents]
                for agent in surviving_agents:
                    agent.resources = initialEndowment
                    agent.old_traces = agent.traces

                new_agents = initializeAgents(numAgents // 2, initialEndowment,
                                              i + 2, surviving_strategies)
                agents = surviving_agents + new_agents
                random.shuffle(agents)

    return {
        "all_donations": all_donations,
        "all_average_final_resources": all_average_final_resources,
        "conditional_survival": conditional_survival,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5:7b-instruct",
                        help="Ollama model id; default qwen2.5:7b-instruct (paper-matched)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", required=True,
                        help="Output file tag, e.g. qwen_s42 → logs/qwen_s42.jsonl")
    parser.add_argument("--num-generations", type=int, default=10,
                        help="Paper Figure-2 default: 10")
    parser.add_argument("--num-agents", type=int, default=12)
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke mode: 2 generations, 4 agents (~3 min)")
    parser.add_argument("--inherit-format", choices=["list", "scored"], default="list",
                        help="Inherited-strategy format (D032 ablation). "
                             "'list' = paper-verbatim list repr (default, baseline). "
                             "'scored' = bullet list with survivor name + final score.")
    args = parser.parse_args()

    if args.smoke:
        args.num_generations = 2
        args.num_agents = 4

    # ---- Configure runtime parameters -----------------------------------
    # No client init — Ollama runs locally and llm_qwen.py imports it directly.
    config.llm = args.model
    config.client = None  # unused for Ollama, kept for API parity
    config.cooperationGain = 2.0
    config.punishmentLoss = 2.0
    config.initial_endowment = 10.0
    config.discounted_value = 0.5
    config.number_of_rounds = 12
    config.include_strategy = True
    config.selection_method = "top"
    config.reputation_mechanism = "three_last_traces"
    config.punishment_mechanism = "none"
    config.inherit_format = args.inherit_format
    config.system_prompt = build_system_prompt(
        config.initial_endowment, config.cooperationGain,
        config.punishment_mechanism, config.punishmentLoss,
    )

    # ---- Seeding --------------------------------------------------------
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ---- Logging --------------------------------------------------------
    log_path = f"logs/{args.tag}.jsonl"
    set_log_path(log_path)
    print(f"Logging to {log_path}")
    print(f"Model: {config.llm}  (Ollama backend, temperature=0.8 paper-matched)")
    print(f"Seed: {args.seed}")
    print(f"Generations: {args.num_generations}, Agents: {args.num_agents}")
    print(f"Inherit format: {config.inherit_format}")
    print()

    log_event(
        type="experiment_meta",
        model=config.llm,
        backend="ollama",
        temperature=0.8,
        inherit_format=config.inherit_format,
        seed=args.seed,
        tag=args.tag,
        num_generations=args.num_generations,
        num_agents=args.num_agents,
        cooperationGain=config.cooperationGain,
        initial_endowment=config.initial_endowment,
        discounted_value=config.discounted_value,
        selection_method=config.selection_method,
        punishment_mechanism=config.punishment_mechanism,
        smoke=bool(args.smoke),
    )

    # ---- Run ------------------------------------------------------------
    simulation_data = SimulationData(hyperparameters={
        "numGenerations": args.num_generations,
        "numAgents": args.num_agents,
        "initialEndowment": config.initial_endowment,
        "selectionMethod": config.selection_method,
        "cooperationGain": config.cooperationGain,
        "include_strategy": config.include_strategy,
        "discountedValue": config.discounted_value,
        "llm": config.llm,
        "reputation_mechanism": config.reputation_mechanism,
        "punishment_mechanism": config.punishment_mechanism,
        "number_of_rounds": config.number_of_rounds,
        "seed": args.seed,
    })

    try:
        metrics = runGenerations(
            args.num_generations, args.num_agents, config.initial_endowment,
            config.selection_method, simulation_data,
        )
        log_event(type="experiment_done", **{k: v for k, v in metrics.items()
                                              if k != "all_donations"})
        print()
        print("=" * 60)
        print(f"DONE. Log: {log_path}")
        print(f"Generations: {args.num_generations}, Agents: {args.num_agents}")
        print(f"Final avg resources by gen: {metrics['all_average_final_resources']}")
    finally:
        close_log()


if __name__ == "__main__":
    main()
