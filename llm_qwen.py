"""promptLLM — Ollama backend for Qwen, drop-in replacement for llm_openai.

Preserves the exact same signature and behaviour as donor_game_openai/llm_openai.py
so the rest of the code path (strategy.py, donation.py) is byte-for-byte the
verbatim paper port. The ONLY thing that differs from the OpenAI sibling repo
is which LLM serves the chat completion.

Paper supports 3 APIs (OpenAI / Anthropic / Google) in cell 18. Their gpt-4o
branch used the OpenAI API default temperature. The paper reports a global
temperature of 0.8 — we set it explicitly here so the Qwen run matches the
paper's stated temperature (and matches what donor_game/llm.py was set to
after the 5/9 default update).

The system_prompt + user prompt structure mirrors cell 18 exactly.

Threading note: donation.py (paper cell 13) calls promptLLM from a
ThreadPoolExecutor. Ollama's local HTTP server serialises requests internally,
so concurrent calls are safe but execute one at a time. That matches what
the paper's gpt-4o branch experienced (OpenAI also rate-limits).
"""
import time

import ollama

import config


# Paper's reported temperature for all LLM calls.
TEMPERATURE = 0.8


def promptLLM(prompt: str, max_retries: int = 3, initial_wait: int = 1,
              timeout: int = None) -> str:
    """Paper signature preserved. Returns the assistant's content string.

    Retries with exponential backoff on any exception (paper cell 18).
    `timeout` is accepted for signature parity with llm_openai.promptLLM but
    Ollama-py does not expose a per-request timeout; the local model is
    fast (~1-2 s/call) so practical timeouts don't matter.
    """
    last_err = None
    for attempt in range(max_retries):
        try:
            response = ollama.chat(
                model=config.llm,
                messages=[
                    {"role": "system", "content": config.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": TEMPERATURE},
            )
            return response["message"]["content"]
        except Exception as e:
            last_err = e
            if attempt == max_retries - 1:
                raise
            wait_time = initial_wait * (2 ** attempt)
            print(f"Error occurred: {type(e).__name__}: {str(e)[:120]}. "
                  f"Retrying in {wait_time}s...")
            time.sleep(wait_time)
    raise Exception(f"Failed after {max_retries} retries; last: {last_err}")
