"""Runtime parameters — replaces the paper's module-level globals.

Identical to donor_game_openai/config.py except `client` is unused (Ollama
needs no client object — `ollama.chat()` is a module-level function).

The paper notebook (cell 21) declares cooperationGain, punishmentLoss,
discounted_value, system_prompt, llm, etc. as Python globals and reads
them inside functions. We put them in one module so the other ports can
`import config` and reference them by attribute. Semantically the same
as paper's globals; just typed and discoverable.

Set ONCE in main.py via init_config() before any LLM call happens.
"""
# Game parameters — paper cell 21 defaults.
cooperationGain: float = 2.0
punishmentLoss: float = 2.0
initial_endowment: float = 10.0
discounted_value: float = 0.5
number_of_rounds: int = 12

# Selection / mechanism flags — paper cell 21.
include_strategy: bool = True
selection_method: str = "top"
reputation_mechanism: str = "three_last_traces"
punishment_mechanism: str = "none"

# LLM backend (set at startup).
llm: str = ""           # e.g. "qwen2.5:7b-instruct"
client = None           # Unused for Ollama (kept for API parity with openai sibling)
system_prompt: str = "" # paper cell 21 f-string

# Per-request timeout in seconds. Unused for Ollama (the python lib does not
# expose per-request timeouts; local model calls are fast). Kept for API
# parity with the openai/claude sibling repos.
llm_request_timeout: int = 300
