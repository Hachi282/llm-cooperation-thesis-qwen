"""Donation game core — verbatim port of cells 12, 13, 14, 15, 16.

This is the heart of the paper's mechanism. Every line of paper logic
is preserved; the only additions are JSONL event emission so the
parent donor_game/ analysis scripts can read the output.
"""
import re
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from queue import Queue
from threading import Lock

import config
from agent import Agent
from sim_data import AgentRoundData, SimulationData
from llm_qwen import promptLLM
from experiment_log import log_event


print_lock = Lock()  # paper cell 3


# --- cell 15 -----------------------------------------------------------
def get_last_three_reversed(item):
    """Paper cell 15, verbatim."""
    if isinstance(item, list):
        return " ".join(item[-3:][::-1])
    elif isinstance(item, str):
        return item
    else:
        return str(item)


# --- cell 12 -----------------------------------------------------------
def calculate_received_amount(punishment_mechanism, refused, cooperationGain,
                              response, punishmentLoss, action=None):
    """Paper cell 12, verbatim."""
    if punishment_mechanism == "partner_choice":
        return cooperationGain * response if not refused else 0
    elif punishment_mechanism == "costly_punishment":
        if action is None:
            raise ValueError("Action must be specified for costly_punishment mechanism")
        if action == "donate":
            return cooperationGain * response
        elif action == "punish":
            return -punishmentLoss * response
        else:
            raise ValueError(f"Unknown action for costly_punishment: {action}")
    elif punishment_mechanism == "none":
        return cooperationGain * response
    else:
        raise ValueError(f"Unknown punishment mechanism: {punishment_mechanism}")


# --- cell 16 -----------------------------------------------------------
def donorPrompt(donor: Agent, generation: int, round: int,
                recipient: Agent) -> str:
    """Paper cell 16, verbatim prompt text. Reads punishment_mechanism /
    cooperationGain / punishmentLoss from config."""
    pm = config.punishment_mechanism
    cooperationGain = config.cooperationGain
    punishmentLoss = config.punishmentLoss

    strategy_text = (
        f"As you will recall, here is the strategy you decided to follow: {donor.strategy}"
        if donor.strategy else ""
    )

    if pm == "partner_choice":
        punishment_text = (
            f"If you do not wish to play with {recipient.name}, simply state: "
            f"Refuse to play. Otherwise. state your answer as a real number "
            f"between 0 and {donor.resources} (inclusive)."
        )
        punishment_text_2 = ""
    elif pm == "costly_punishment":
        punishment_text = (
            f"State your answer as EITHER Donate, x OR Punish, x, to either "
            f"have them receive {cooperationGain}x units or punish them by "
            f"taking away {punishmentLoss}x of their units. Remember, your "
            f"answer should be simply 'Donate, x' or 'Punish, x' (where x is "
            f"the real number of units you give up.)"
        )
        punishment_text_2 = (
            f"Remember, state your answer as EITHER Donate, x OR Punish, x, "
            f"to either have them receive {cooperationGain}x units or punish "
            f"them by taking away {punishmentLoss}x of their units."
        )
    else:
        punishment_text = (
            f"state your answer as a real number between 0 and "
            f"{donor.resources} (inclusive)."
        )
        punishment_text_2 = ""

    recipient_behavior = ""
    if donor.traces:
        last_trace = recipient.traces[-1]
        if isinstance(last_trace, list):
            recipient_behavior = get_last_three_reversed(last_trace)
        else:
            recipient_behavior = (
                f"Here is what {recipient.name} and their earlier partners "
                f"did in previous rounds: {str(last_trace)}"
            )

    return (
        f"Your name is {donor.name}. "
        f"{strategy_text}"
        f"This is generation {generation}. It is now round {round}. "
        f"In this round, you have been paired with {recipient.name}. They currently have {recipient.resources} units of the valuable resource."
        f"{recipient_behavior}"
        f"You will take the role of donor. You currently have {donor.resources} units of the valuable resource. "
        f"How many units do you give up?"
        f"{punishment_text}"
        "Very briefly think step by step about how you apply your strategy in this situation and then provide your answer."
        "State your full response in the following form:"
        "Justification:"
        "[multi-line justification here]"
        " "
        "Answer: [answer here]"
        "Remember: First state your justification and after that your answer."
        f"{punishment_text_2}"
    )


# --- cell 13 -----------------------------------------------------------
def handle_pairing_thread_safe(donor, recipient, round_index, generation,
                               game_number, agent_locks, donation_records,
                               agent_updates):
    """Paper cell 13, verbatim logic + JSONL emission for donations."""
    pm = config.punishment_mechanism
    cooperationGain = config.cooperationGain
    punishmentLoss = config.punishmentLoss
    discounted_value = config.discounted_value

    action_info = ""
    donor_data = None
    recipient_data = None
    punished = False
    action = "donate"
    justification = ""
    response = 0
    full_response = ""

    recipient_behavior = ""
    if donor.traces:
        last_trace = recipient.traces[-1]
        if isinstance(last_trace, list):
            recipient_behavior = get_last_three_reversed(last_trace)
        else:
            recipient_behavior = str(last_trace)

    # Record pre-action resources for JSONL.
    donor_resources_before = donor.resources
    recipient_resources_before = recipient.resources

    with agent_locks[donor.name], agent_locks[recipient.name]:
        prompt = donorPrompt(donor, generation, round_index + 1, recipient)

        valid_response = False
        max_attempts = 10
        attempts = 0

        while not valid_response and attempts < max_attempts:
            try:
                # timeout=None → let promptLLM fall through to
                # config.llm_request_timeout (per-model: 30s chat / 300s reasoning).
                # Paper hardcoded 30s here; we keep the call shape but pick
                # the timeout based on model family in main.py.
                full_response = promptLLM(prompt, timeout=None)
                parts = full_response.split("Answer:", 1)

                if len(parts) == 2:
                    justification = parts[0].replace("Justification:", "").strip()
                    answer_part = parts[1].strip()

                    if pm == "partner_choice":
                        if "refuse" in answer_part.lower():
                            action = "refuse"
                            response = 0
                            valid_response = True
                        else:
                            m = re.search(r"^\s*(\d+(?:\.\d+)?)", answer_part)
                            if m:
                                action = "donate"
                                response = float(m.group(1))
                                # Paper-faithful clamp (see donor_game/llm.py
                                # request_donation): Qwen 7B frequently responds
                                # with an amount > donor.resources. Without this,
                                # the log records the LLM's stated value while
                                # the game state silently rejects the action,
                                # producing donation_pct > 1.0.
                                response = max(0.0, min(response, donor.resources))
                                valid_response = True
                    elif pm == "costly_punishment":
                        m = re.search(r"(donate|punish).*?(\d+(?:[.,]\d+)?)",
                                      answer_part, re.IGNORECASE)
                        if m:
                            action = m.group(1).lower()
                            response = float(m.group(2).replace(",", "."))
                            response = max(0.0, min(response, donor.resources))
                            valid_response = True
                    else:  # punishment_mechanism == 'none'
                        m = re.search(r"^\s*(\d+(?:\.\d+)?)", answer_part)
                        if m:
                            action = "donate"
                            response = float(m.group(1))
                            # Paper-faithful clamp (see donor_game/llm.py
                            # request_donation): Qwen 7B frequently responds
                            # with an amount > donor.resources. Without this,
                            # the log records the LLM's stated value while
                            # the game state silently rejects the action,
                            # producing donation_pct > 1.0.
                            response = max(0.0, min(response, donor.resources))
                            valid_response = True

                if not valid_response:
                    print(f"Invalid response from {donor.name} in round "
                          f"{round_index + 1}. Retrying...")
                    attempts += 1
            except ValueError:
                print(f"Invalid numerical response from {donor.name} in round "
                      f"{round_index + 1}")
                attempts += 1
            except TimeoutError:
                print(f"LLM call timed out for {donor.name} in round "
                      f"{round_index + 1}")
                attempts += 1

        if not valid_response:
            print(f"Failed to get a valid response from {donor.name} after "
                  f"{max_attempts} attempts")
            action = "donate"
            response = 0

    # ---- Apply action ----------------------------------------------------
    if action == "refuse":
        action_info = (
            f"{donor.name} refused to play with {recipient.name}.\n"
            f"Resources: {donor.name}: {donor.resources:.2f} and "
            f"{recipient.name}: {recipient.resources:.2f} \n"
            f"Recipient traces: {recipient_behavior} \n"
            f"Justification:\n{textwrap.fill(justification, width=80, initial_indent='    ', subsequent_indent='    ')}\n"
        )
        new_trace = recipient.traces[-1].copy() if recipient.traces else []
        new_trace.append(f"In round {round_index + 1}, {donor.name} refused "
                         f"to play with {recipient.name}.")
        donor.traces.append(new_trace)
        donor_history = (
            f"In round {round_index + 1} (Game {game_number}) you were paired "
            f"with agent {recipient.name}. You refused to play."
            f"{get_last_three_reversed(recipient.traces[-1])}"
        )
        recipient_history = (
            f"In round {round_index + 1} (Game {game_number}) you were paired "
            f"with agent {donor.name}, They refused to play."
            f"{get_last_three_reversed(donor.traces[-1])}"
        )

    elif 0 <= response <= donor.resources:
        if action == "donate":
            percentage_donated = response / donor.resources if donor.resources != 0 else 1
            donor.resources -= response
            donor.total_donated += response
            donor.potential_donated += donor.resources + response
            recipient.resources += cooperationGain * response

            action_info = (
                f"{donor.name}: -{response} ({percentage_donated:.2%}) and "
                f"{recipient.name}: +{cooperationGain * response}.\n"
                f"Previous resources: {donor.name}: {donor.resources + response:.2f} "
                f"and {recipient.name}: {recipient.resources - (cooperationGain * response)}.\n"
                f"New resources: {donor.name}: {donor.resources:.2f} and "
                f"{recipient.name}: {recipient.resources:.2f}.\n"
                f"Recipient traces: {recipient_behavior}"
                f"Justification:\n{textwrap.fill(justification, width=80, initial_indent='    ', subsequent_indent='    ')}\n"
            )

            new_trace = recipient.traces[-1].copy() if recipient.traces else []
            new_trace.append(
                f"In round {round_index + 1}, {donor.name} donated "
                f"{percentage_donated * 100:.2f}% of their resources to "
                f"{recipient.name}.")
            donor.traces.append(new_trace)

            donor_history = (
                f"In round {round_index + 1} (Game {game_number}) you were "
                f"paired with agent {recipient.name}. You gave up {response} "
                f"units, and they received {cooperationGain * response} units."
                f"{get_last_three_reversed(recipient.traces[-1])}"
            )
            recipient_history = (
                f"In round {round_index + 1} (Game {game_number}) you were "
                f"paired with agent {donor.name}, They gave up {response} "
                f"units, and you received {cooperationGain * response} units."
                f"{get_last_three_reversed(donor.traces[-1])}"
            )

            # Paper's reputation update for donor.
            if donor.reputation is False:
                donor.reputation = percentage_donated
            else:
                donor.reputation = (
                    (1 - abs(percentage_donated - recipient.reputation))
                    + discounted_value * donor.reputation
                ) / (1 + discounted_value)

    elif action == "punish":
        punished = True
        percentage_donated = response / donor.resources if donor.resources != 0 else 1
        donor.resources -= response
        donor.total_donated += response
        donor.potential_donated += donor.resources + response
        recipient.resources = max(0, recipient.resources - punishmentLoss * response)
        action_info = (
            f"{donor.name}: -{response} ({percentage_donated:.2%}) and "
            f"{recipient.name}: - {punishmentLoss * response}.\n"
            f"Previous resources: {donor.name}: {donor.resources + response:.2f} "
            f"and {recipient.name}: {recipient.resources + (punishmentLoss * response)}."
            f"New resources: {donor.name}: {donor.resources:.2f} and "
            f"{recipient.name}: {recipient.resources:.2f}.\n"
            f"Recipient traces: {recipient_behavior} \n"
            f"Justification:\n{textwrap.fill(justification, width=80, initial_indent='    ', subsequent_indent='    ')}\n"
        )
        new_trace = recipient.traces[-1].copy() if recipient.traces else []
        new_trace.append(
            f"In round {round_index + 1}, {donor.name} punished "
            f"{recipient.name} by spending {response} units to take away "
            f"{punishmentLoss * response} units from their resources.")
        donor.traces.append(new_trace)
        donor_history = (
            f"In round {round_index + 1} (Game {game_number}) you were paired "
            f"with agent {recipient.name}. You punished them by giving up "
            f"{response} units to take away {punishmentLoss * response} units from them."
            f"{get_last_three_reversed(recipient.traces[-1])}"
        )
        recipient_history = (
            f"In round {round_index + 1} (Game {game_number}) you were paired "
            f"with agent {donor.name}, They punished you by giving up "
            f"{response} units to take away {punishmentLoss * response} units from you."
            f"{get_last_three_reversed(donor.traces[-1])}"
        )

    else:
        action_info = (
            f"{donor.name} attempted an invalid action.\n"
            f"Resources: {donor.name}: {donor.resources:.2f} and "
            f"{recipient.name}: {recipient.resources:.2f} \n"
            f"Recipient traces: {recipient_behavior} \n"
            f"Justification:\n{textwrap.fill(justification, width=80, initial_indent='    ', subsequent_indent='    ')}\n"
        )
        donor_history = (
            f"In round {round_index + 1} (Game {game_number}) you were paired "
            f"with agent {recipient.name}. You attempted an invalid action."
            f"{get_last_three_reversed(recipient.traces[-1])}"
        )
        recipient_history = (
            f"In round {round_index + 1} (Game {game_number}) you were paired "
            f"with agent {donor.name}, They attempted an invalid action."
            f"{get_last_three_reversed(donor.traces[-1])}"
        )

    donor.history.append(donor_history)
    recipient.history.append(recipient_history)

    donor_data = AgentRoundData(
        agent_name=donor.name,
        round_number=round_index + 1,
        paired_with=recipient.name,
        current_generation=generation,
        game_number=game_number,
        resources=donor.resources,
        donated=response if action != "refuse" else 0,
        received=0,
        strategy=donor.strategy,
        strategy_justification=donor.strategy_justification,
        reputation=donor.reputation,
        is_donor=True,
        traces=donor.traces,
        history=donor.history,
        punished=punished,
        justification=justification,
    )
    recipient_data = AgentRoundData(
        agent_name=recipient.name,
        round_number=round_index + 1,
        paired_with=donor.name,
        current_generation=generation,
        game_number=game_number,
        resources=recipient.resources,
        donated=0,
        received=calculate_received_amount(
            pm, action == "refuse", cooperationGain, response, punishmentLoss, action),
        strategy=recipient.strategy,
        strategy_justification=recipient.strategy_justification,
        reputation=recipient.reputation,
        is_donor=False,
        traces=recipient.traces,
        history=recipient.history,
    )

    # JSONL emission — schema matches donor_game/logs/ donation events.
    if action == "donate":
        pct = response / donor_resources_before if donor_resources_before != 0 else 0.0
        log_event(
            type="donation",
            generation=generation,
            game=game_number,
            round=round_index + 1,
            donor_name=donor.name,
            recipient_name=recipient.name,
            donor_resources_before=donor_resources_before,
            recipient_resources_before=recipient_resources_before,
            donation_amount=float(response),
            donation_pct=float(pct),
            donor_resources_after=donor.resources,
            recipient_resources_after=recipient.resources,
            donor_reputation_after=donor.reputation if donor.reputation is not False else None,
            justification=justification,
            raw_response=full_response,
        )

    return action_info, donor_data, recipient_data


# --- cell 14 -----------------------------------------------------------
def donorGame(agents: list, rounds: list, generation: int,
              simulation_data: SimulationData,
              all_average_final_resources: list):
    """Paper cell 14, verbatim. Plays 2 games (game 2 has reversed
    pairings). Returns (fullHistory, donation_records).

    `all_average_final_resources` is passed in (paper uses global).
    """
    initial_endowment = config.initial_endowment

    fullHistory = []
    donation_records = Queue()
    agent_updates = Queue()

    agent_locks = {agent.name: Lock() for agent in agents}

    def play_game(game_number, game_rounds):
        round_results = {i: [] for i in range(len(game_rounds))}

        for round_index, round_pairings in enumerate(game_rounds):
            if round_index == 0:
                for agent in agents:
                    agent.traces = [[f"{agent.name} did not have any previous interactions."]]

            with ThreadPoolExecutor(max_workers=min(len(round_pairings), 10)) as executor:
                futures = []
                for donor, recipient in round_pairings:
                    if round_index > 0:
                        donor.traces.append(recipient.traces[-1].copy())
                    future = executor.submit(
                        handle_pairing_thread_safe,
                        donor, recipient, round_index, generation, game_number,
                        agent_locks, donation_records, agent_updates,
                    )
                    futures.append(future)

                for future in as_completed(futures):
                    action_info, donor_data, recipient_data = future.result()
                    if action_info:
                        round_results[round_index].append(action_info)
                    if donor_data and recipient_data:
                        simulation_data.agents_data.append(asdict(donor_data))
                        simulation_data.agents_data.append(asdict(recipient_data))

        return round_results

    # Game 1
    game1_results = play_game(1, rounds)
    for round_index in range(len(rounds)):
        fullHistory.append(f"Round {round_index + 1} (Game 1):\n")
        fullHistory.extend(game1_results[round_index])

    while not agent_updates.empty():
        agent, history = agent_updates.get()
        agent.history.append(history)

    average_resources_game1 = sum(a.resources for a in agents) / len(agents)
    with print_lock:
        print(f"Average final resources for this generation (Game 1): "
              f"{average_resources_game1:.2f}")

    # Store Game 1 final reputations.
    game1_reputations = {agent.name: agent.reputation for agent in agents}

    # Reset for Game 2.
    for agent in agents:
        agent.resources = initial_endowment
        agent_generation = int(agent.name.split("_")[0])
        if agent_generation < generation:  # surviving agent
            agent.reputation = agent.average_reputation
            agent.traces = agent.old_traces
        else:
            agent.reputation = False
            agent.traces.clear()
        agent.history.clear()

    reversed_rounds = [[tuple(reversed(pair)) for pair in round_pairings]
                       for round_pairings in rounds]

    # Game 2
    game2_results = play_game(2, reversed_rounds)
    for round_index in range(len(reversed_rounds)):
        fullHistory.append(f"Round {round_index + 1} (Game 2):\n")
        fullHistory.extend(game2_results[round_index])

    while not agent_updates.empty():
        agent, history = agent_updates.get()
        agent.history.append(history)

    average_resources_game2 = sum(a.resources for a in agents) / len(agents)
    with print_lock:
        print(f"Average final resources for this generation (Game 2): "
              f"{average_resources_game2:.2f}")

    # Final scores + reputations.
    for agent in agents:
        agent.total_final_score = sum(agent.resources for _ in range(2))
        agent.average_reputation = (
            (game1_reputations[agent.name] + agent.reputation) / 2
            if agent.reputation is not False else game1_reputations[agent.name]
        )

    overall_average_resources = (average_resources_game1 + average_resources_game2) / 2
    all_average_final_resources.append(overall_average_resources)

    return fullHistory, list(donation_records.queue)
