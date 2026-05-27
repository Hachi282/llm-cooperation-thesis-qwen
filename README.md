# donor_game_qwen — Paper-faithful Qwen run (sibling to donor_game_openai)

Third sibling folder. **Same verbatim paper port** as `donor_game_openai/`,
**only LLM client swapped** to Ollama for local Qwen 2.5 7B.

## Why this exists (not donor_game/)

`donor_game/` is the user's own framework, 6 paper-deviation patches applied.
Game logic is behaviourally paper-faithful but the CODE PATH (game.py 295 L,
evolution.py 110 L, separate trace.py, mechanism_classifier.py) is not a
verbatim port. For the RF5 cross-model validation we need **identical code
path** across Qwen / gpt-5 / Claude — so this repo ports the openai verbatim
notebook port and just swaps the LLM backend.

## File-by-file relationship to donor_game_openai

| File | Status |
|---|---|
| `agent.py` | **identical copy** |
| `sim_data.py` | **identical copy** |
| `pairing.py` | **identical copy** |
| `selection.py` | **identical copy** |
| `experiment_log.py` | **identical copy** |
| `strategy.py` | identical except `from llm_openai → from llm_qwen` |
| `donation.py` | identical except `from llm_openai → from llm_qwen` |
| `config.py` | identical except comment + `llm_request_timeout` default |
| `main.py` | identical structure; OpenAI client init dropped, --model default = Qwen |
| `llm_qwen.py` | **NEW** — Ollama-backed `promptLLM`, signature parity with `llm_openai.py` |

## Setup

```bash
# Ollama running locally, Qwen 2.5 7B pulled:
ollama pull qwen2.5:7b-instruct
pip install -r requirements.txt
```

## Running

```bash
# Smoke run (~3 min): 2 generations × 4 agents
python main.py --seed 42 --tag smoke --smoke

# Full single-seed run (~45-60 min on a decent GPU)
python main.py --seed 42 --tag qwen_s42
```

Outputs land in `logs/<tag>.jsonl`.

## Temperature

Paper reports T=0.8 globally. We set it explicitly in `llm_qwen.py`
(unlike `llm_openai.py` which leaves it to OpenAI's API default — the
paper's gpt-4o branch did the same). This is the principled deviation
from "byte-identical": we want the **same temperature** as paper across
all three backends.

## Analyzing

Use parent folder analysis scripts; the JSONL schema matches `donor_game/logs/`
and `donor_game_openai/logs/`.
