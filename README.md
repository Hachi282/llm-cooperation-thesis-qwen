# donor_game_qwen — Qwen v2 (paper-verbatim port + Ollama Qwen)

Third sibling repo. **Same near-verbatim port** as
[`donor_game_openai/`](https://github.com/Hachi282/llm-cooperation-thesis-openai),
**only the LLM client swapped** to Ollama for local Qwen 2.5 7B. Created 5/28
specifically for **D032** — a clean cross-model attractor comparison that needs
the Qwen code path to be byte-identical with the OpenAI sibling (which the
parent repo's "Qwen v1" framework is not, even after its 6 paper-faithfulness
patches).

This repo is one of three sibling repos:

| Repo | What it is | LLM backend | Role |
|---|---|---|---|
| [`Hachi282/llm-cooperation-thesis`](https://github.com/Hachi282/llm-cooperation-thesis) | Qwen v1 framework — user's own implementation | Ollama + qwen2.5:7b-instruct | Main analysis hub, lab notebook, thesis notes |
| [`Hachi282/llm-cooperation-thesis-openai`](https://github.com/Hachi282/llm-cooperation-thesis-openai) | Paper-verbatim port + OpenAI ladder | OpenAI API (gpt-4o / gpt-5-mini / gpt-5) | Cross-capability comparison + thesis figures |
| **`donor_game_qwen/`** (this one, local-only — no GitHub remote yet) | Same verbatim port + Ollama Qwen | Ollama + qwen2.5:7b-instruct | **"Qwen v2" — D032 controlled ablation** |

> ⚠️ **No GitHub remote yet.** This repo is git-tracked locally
> (`git init` 5/28). To publish: `gh repo create
> Hachi282/llm-cooperation-thesis-qwen --public --source=. --push`. Pending a
> separate decision on whether to publish (the main thesis findings already
> live across the other two public repos).

---

## Why this repo exists

The parent repo (`donor_game/`) is "Qwen v1" — the user's own framework with
six paper-faithfulness patches applied (see A002 in parent's `notes/experiments.md`).
Behaviour is paper-faithful but the **code path** is not a verbatim notebook
port: `game.py` is 295 lines, `evolution.py` is 110, there's a separate
`trace.py`, and a `mechanism_classifier.py` that the paper doesn't have.

For **D031** (5/28) we wanted to test whether the "Qwen multimodality" finding
in E004 was a property of the framework, the model, or both. The only honest
test is a cross-model comparison with **identical code path** — so we ported
the `donor_game_openai/` modules verbatim and swapped `llm_openai.py` for
`llm_qwen.py` (Ollama backend, same `promptLLM` signature).

D031 found: clean Qwen v2 is **single-attractor**. Five seeds, per-gen
r = 0.073 ± 0.033, no outlier. That overturned the "multi-attractor across
Qwen + gpt-5" provisional headline from the 5/22 advisor meeting.

**D032** (5/28-29) then used this repo to run the pre-registered
scored-inheritance ablation: take the same code, change *only* the
inherited-strategy format from list-repr to `\n- {name} (final score {score}): {strategy}`
bullets (`--inherit-format scored`), keep everything else identical. Five
seeds. Five pre-registered decision-rule criteria. Result: all five fail.
Multi-attractor did not return. Capability-gated interpretation supported.

---

## File-by-file relationship to `donor_game_openai/`

| File | Status vs OpenAI sibling |
|---|---|
| `agent.py` | **identical copy** |
| `sim_data.py` | **identical copy** |
| `pairing.py` | **identical copy** |
| `selection.py` | **identical copy** |
| `experiment_log.py` | **identical copy** |
| `strategy.py` | identical except `from llm_openai → from llm_qwen` |
| `donation.py` | identical except `from llm_openai → from llm_qwen` and the **D029 clamp** (mirror of OpenAI's commit `08620b9`) |
| `config.py` | identical structure; `llm_request_timeout` adjusted for local model |
| `main.py` | identical structure; OpenAI client init dropped, `--model` defaults to `qwen2.5:7b-instruct`, **`--inherit-format {list,scored}` added for D032** |
| `llm_qwen.py` | **NEW** — Ollama-backed `promptLLM`, signature parity with `llm_openai.py`; temperature **explicitly set to 0.8** (paper-matched) |
| `run_5_seeds.sh` | NEW — sequential 5-seed launcher (Ollama serialises so parallel doesn't help) |
| `run_5_seeds_scored.sh` | NEW — D032 ablation launcher |

The only deliberate behavioural difference from `donor_game_openai/`: this repo
sets `temperature=0.8` explicitly in `llm_qwen.py` (the OpenAI port leaves it
to the API default because gpt-5 reasoning models ignore it anyway). The paper
reports `T=0.8` globally; we want all three backends to actually hit that.

---

## Setup

```bash
# Ollama running locally, Qwen 2.5 7B pulled
ollama pull qwen2.5:7b-instruct
pip install -r requirements.txt    # ollama, numpy, scipy, pandas
```

## Running

```bash
# Smoke (~3 min): 2 generations × 4 agents, baseline list inheritance
python -X utf8 main.py --seed 42 --tag smoke --smoke

# Full single-seed, list inheritance (baseline) — ~45-60 min on a decent GPU
python -X utf8 main.py --seed 42 --tag qwen_v2_s42

# Full single-seed, scored inheritance (D032 ablation)
python -X utf8 main.py --seed 42 --tag qwen_v2_scored_s42 --inherit-format scored

# All 5 baseline seeds sequentially (Ollama serialises) — ~4-5 hr total
bash run_5_seeds.sh

# All 5 D032 ablation seeds sequentially
bash run_5_seeds_scored.sh
```

Outputs land in `logs/<tag>.jsonl`. `logs/` is gitignored.

## The `--inherit-format` flag (D032 core variable)

This is the only knob that distinguishes Qwen v2 baseline from the D032 ablation:

| value | what gets passed to gen 2+ `generate_strategy()` |
|---|---|
| **`list`** (default, paper-verbatim) | `['My strategy will be ...', 'My strategy will be ...']` — raw list of strategy strings, Python `repr` interpolation, no names, no scores |
| **`scored`** (D032 ablation) | `\n- 1_3 (final score 1071.8): My strategy will be ...\n- 1_11 (final score 1014.6): My strategy will be ...` — bullet list with survivor name + final score (mirrors `donor_game/llm.py`'s format) |

The paper notebook does the former (`list`); Qwen v1 does the latter (`scored`).
D032 asked: was Qwen v1's apparent multimodality driven by score-visible
inheritance? Answer: no. See parent repo D032/D033.

---

## Output schema

JSONL events match `donor_game_openai/` exactly (intentional, so the parent
repo's analysis scripts can read both without changes). One addition: the
`experiment_meta` event records `backend="ollama"`, `temperature=0.8`, and
`inherit_format` (one of `list` or `scored`):

```json
{"ts": "2026-05-28T15:40:01", "type": "experiment_meta",
 "model": "qwen2.5:7b-instruct", "backend": "ollama", "temperature": 0.8,
 "inherit_format": "scored", "seed": 42, "tag": "qwen_v2_scored_s42",
 "num_generations": 10, "num_agents": 12, ...}
```

## Analyzing results

Use the parent repo's analysis scripts (they read JSONL by tag pattern):

```bash
cd ../donor_game
python compute_canonical_metrics.py    # canonical r / donation table
python make_d033_figures.py             # 4 thesis figures
```

---

## What's in this repo's logs (canonical D031/D032 set)

| Tag | Inherit format | Seed | Per-gen r | Mean donation | Used in |
|---|---|---:|---:|---:|---|
| `qwen_v2_s7` | list | 7 | 0.048 | 41.7% | D031 baseline |
| `qwen_v2_s13` | list | 13 | 0.045 | 42.5% | D031 baseline |
| `qwen_v2_s42` | list | 42 | 0.124 | 44.1% | D031 baseline |
| `qwen_v2_s99` | list | 99 | 0.062 | 46.5% | D031 baseline |
| `qwen_v2_s2024` | list | 2024 | 0.085 | 45.0% | D031 baseline |
| `qwen_v2_scored_s7` | scored | 7 | 0.049 | 48.9% | **D032 ablation** |
| `qwen_v2_scored_s13` | scored | 13 | 0.070 | 41.8% | **D032 ablation** |
| `qwen_v2_scored_s42` | scored | 42 | 0.010 | 50.7% | **D032 ablation** |
| `qwen_v2_scored_s99` | scored | 99 | 0.053 | 52.3% | **D032 ablation** |
| `qwen_v2_scored_s2024` | scored | 2024 | 0.031 | 48.2% | **D032 ablation** |

Aggregates:

- **list**: per-gen r = 0.073 ± 0.033, range 0.079, single attractor
- **scored**: per-gen r = 0.043 ± 0.023, range 0.060, single attractor
- D032 decision-rule check: **5/5 fail** → multi-attractor did not return

---

## Weekly progress log

Quick-scan timeline of work in this repo. Full reasoning in parent repo's
[`notes/lab_notebook.md`](https://github.com/Hachi282/llm-cooperation-thesis/blob/main/notes/lab_notebook.md).
Current week is **bold**.

### 2026-05-28 — Repo created, baseline run, clamp bug discovered

- 5/28 morning: **D028** decision to open this repo. Verbatim port from
  `donor_game_openai/`; only `llm_openai.py` replaced with `llm_qwen.py`;
  smoke test passes
- First 5-seed baseline run launched (~4-5 hr sequential)
- Analysis revealed **`donation_pct > 1.0` events** — 15.6% of seed 7. Traced
  to the paper-inherited silent logging bug (V&H's notebook doesn't clamp
  LLM responses to `donor.resources`); capable OpenAI models never tripped
  it, Qwen 7B does
- **D029** patch applied: `response = max(0.0, min(response, donor.resources))`
  added in both `donor_game_openai/donation.py` (commit `08620b9`) and this
  repo's `donation.py` (local commit `ebc9adc`)
- 5-seed baseline re-run with the clamp fix. All clean (`donation_pct` max
  = 1.000, zero `>1.0` events)
- **D031** decided: clean Qwen v2 is **single-attractor**. 5 seeds, per-gen
  r = 0.073 ± 0.033, no outlier. Overturns the 5/22 cross-model
  multi-attractor headline.
- **D032** pre-registered: scored-inheritance ablation. Five decision-rule
  criteria committed in advance to prevent post-hoc eyeballing.
- Implemented `--inherit-format {list, scored}` flag (local commit `d1e557d`).
  Smoke-tested both modes (`list` produces `['...']` repr, `scored`
  produces `- 1_3 (final score 108.0): ...` bullets — both with
  clamp intact)
- D032 5-seed run launched

### 2026-05-29 — D032 result

- D032 5-seed scored run finished. Analysis: per-gen r = 0.043 ± 0.023,
  range 0.060. All 5 pre-registered decision-rule criteria fail. Donation
  rose 44%→50% but reciprocity dropped from 0.073 to 0.043
- **D033**: capability-gated interpretation strongly supported; transmission-
  information confound ruled out. Side observation: performance signal
  (visible scores) makes Qwen *more generous* but *less reciprocal* —
  reinforces the original rhetoric-behaviour gap from the parent repo's
  Pilot A/C

### 2026-05-30 (current) — Thesis figures use this repo's data

- Master cross-model table built using `donor_game/compute_canonical_metrics.py`
  reading this repo's `qwen_v2_*` and `qwen_v2_scored_*` logs
- 4 thesis figures generated by `donor_game/make_d033_figures.py` — three of
  them (F1 regime overview, F2 per-gen r, F3 D032 ablation, F4 attractor map)
  rely on this repo's data
- Decision **D035**: the 5/22 → 5/30 narrative shift (cross-model
  multi-attractor → capability-gated) is written into the thesis as a
  Turpin-grade critique-and-replace methodology *feature*, not a retrofit

### Coming up

- Possibly a Claude port (`donor_game_claude/`) using this same template — same
  modular structure, swap the LLM client. Pending advisor approval of the
  Claude run plan
- Optional: seeds 6–10 robustness check on `qwen_v2_*` to head off any
  "n=5 not enough" challenge

---

## Key cross-repo links

- **Decision history** (D001–D035): [parent repo `notes/lab_notebook.md`](https://github.com/Hachi282/llm-cooperation-thesis/blob/main/notes/lab_notebook.md)
- **1-page thesis ground truth**: [parent repo `notes/thesis_spine.md`](https://github.com/Hachi282/llm-cooperation-thesis/blob/main/notes/thesis_spine.md)
- **Research question chain v1 → v3**: [parent repo `notes/research_question.md`](https://github.com/Hachi282/llm-cooperation-thesis/blob/main/notes/research_question.md)
- **A002 deviation audit (#7-9 incl. clamp, system-prompt placeholder, inheritance format)**: [parent repo `notes/experiments.md`](https://github.com/Hachi282/llm-cooperation-thesis/blob/main/notes/experiments.md)
- **OpenAI sibling repo**: [Hachi282/llm-cooperation-thesis-openai](https://github.com/Hachi282/llm-cooperation-thesis-openai)
- **Thesis figures** (F1-F4): [openai repo `figures/thesis/`](https://github.com/Hachi282/llm-cooperation-thesis-openai/tree/main/figures/thesis)
