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
| **[`Hachi282/llm-cooperation-thesis-qwen`](https://github.com/Hachi282/llm-cooperation-thesis-qwen)** (this one) | Same verbatim port + Ollama Qwen | Ollama + qwen2.5:7b-instruct | **"Qwen v2" — D032 controlled ablation** |

---

## 為什麼有這個 repo（與目前論文狀態）

Parent repo（`donor_game/`）是「Qwen v1」—— user 自己寫的 framework、應用了 6 個
paper-faithfulness patches（見 parent repo `notes/experiments.md` 的 A002）。
**行為層**對齊 paper、但 **code path** 不是 verbatim notebook port：`game.py`
有 295 行、`evolution.py` 110 行、有獨立的 `trace.py`、還有一個 paper 沒有的
`mechanism_classifier.py`。

**D031**（5/28）想驗證：E004 看到的「Qwen 多峰」是 framework artifact、是 model
特性、還是兩者交互？唯一誠實的測試方式是 **code path 完全一致** 的 cross-model
對比 —— 所以把 `donor_game_openai/` 整套 module 字字 verbatim 搬過來、只把
`llm_openai.py` 換成 `llm_qwen.py`（Ollama backend、同樣的 `promptLLM` 介面）。

**D031 結果**：乾淨的 Qwen v2 是**單峰**。5 seed、per-gen r = 0.073 ± 0.033、
無 outlier。**5/22 教授 meeting 上的「multi-attractor 跨 Qwen + gpt-5」headline
被推翻。**

**D032**（5/28–29）接著用這個 repo 跑 pre-registered 的 scored-inheritance
ablation：同樣的 code、**只改 inherited-strategy 格式一個變數**（list repr →
`\n- {name} (final score {score}): {strategy}` bullet、`--inherit-format scored`），
其他完全相同。5 seed、5 條 pre-registered decision-rule criteria。**結果：5 條
全 fail。** 多峰沒回來、**capability-gated interpretation 強烈支持**（strongly
supports，未證明；Claude 3.5 Sonnet 是計畫中的 out-of-sample test）。

論文現在的 working main claim、完整 hypothesis 鏈、caveats 全部見 parent repo
`donor_game/notes/thesis_spine.md`。

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

## 每週進度（Weekly progress log）

這個 repo 每週做了什麼的快速時間線。完整推理請看 parent repo 的
[`notes/lab_notebook.md`](https://github.com/Hachi282/llm-cooperation-thesis/blob/main/notes/lab_notebook.md)。
**粗體**為本週。

### 2026-05-28 — Repo 建立、baseline 跑完、抓到 clamp bug

- 5/28 早上：**D028 決定**開這個 repo。verbatim port 從 `donor_game_openai/`、
  只換 `llm_openai.py → llm_qwen.py`、smoke test 通過
- 第一次 5-seed baseline run 啟動（~4-5 hr 序列跑、Ollama 內部會序列化）
- 分析時發現 **`donation_pct > 1.0` 的事件** —— seed 7 上 15.6% events 中招。追到
  paper 自己的 silent logging bug（V&H notebook 沒把 LLM 回答 clamp 到
  `donor.resources`）；capable OpenAI model 都不會踩、Qwen 7B 會
- **D029 patch**：加 `response = max(0.0, min(response, donor.resources))` 到
  `donor_game_openai/donation.py`（commit `08620b9`）跟本 repo 的
  `donation.py`（local commit `ebc9adc`）
- 帶 clamp fix 重跑 5-seed baseline。全乾淨（`donation_pct` max = 1.000、0 個
  >1.0 事件）
- **D031 結論**：乾淨的 Qwen v2 是**單峰**。5 seed、per-gen r = 0.073 ± 0.033、
  無 outlier。**5/22 跨模型 multi-attractor headline 被推翻**
- **D032 pre-register**：scored-inheritance ablation。5 條 decision-rule
  criteria 跑前就釘住、防事後肉眼判讀
- 實作 `--inherit-format {list, scored}` flag（local commit `d1e557d`）。兩個
  mode 都 smoke 過（`list` 產生 `['...']` repr、`scored` 產生
  `- 1_3 (final score 108.0): ...` bullet、clamp 都生效）
- D032 5-seed run 啟動

### 2026-05-29 — D032 結果出爐

- D032 scored 5-seed 跑完。分析結果：per-gen r = 0.043 ± 0.023、range 0.060。
  **5 條 pre-registered decision-rule criteria 全 fail**。donation 從 44% 升到
  50%、但 reciprocity 從 0.073 跌到 0.043
- **D033 結論**：capability-gated interpretation **strongly supported**；
  transmission-information confound 被排除。次發現：performance signal
  （可見 score）讓 Qwen **更慷慨但更不互惠** —— 印證 parent repo Pilot A/C 的
  rhetoric-behaviour gap

### 2026-05-30（本週）— 論文圖用了這個 repo 的資料

- Master cross-model 表用 `donor_game/compute_canonical_metrics.py` 重算、讀本
  repo 的 `qwen_v2_*` 跟 `qwen_v2_scored_*` log
- 4 張核心論文圖由 `donor_game/make_d033_figures.py` 產出 ——
  其中 F1 regime overview、F2 per-gen r、F3 D032 ablation、F4 attractor map
  都用到本 repo 的資料
- **D035 決定**：5/22 → 5/30 的 narrative shift（cross-model multi-attractor
  → capability-gated）寫進論文時 **framed as a feature, not retrofit** ——
  展現 Turpin-grade critique-and-replace methodology

### 接下來

- **✅ 已完成（D039–D043）**：Claude out-of-sample 測試已跑——但不是新開 `donor_game_claude/`，而是用 `donor_game_openai/` port 經 gateway 跑 `claude-sonnet-5` / `claude-opus-4-8` 各 **n=20**。結果**推翻 capability-gated**，改為 family/model-specific + 兩軸脫鉤(見 parent lab_notebook)。
- **⚠️ 進行中（D053）**：gemma4 out-of-sample(新 frontier family、教授指示)亦走 `donor_game_openai/` port + vLLM,非本 repo。
- 可選(仍未做)：補 `qwen_v2_*` seed 6–10 robustness check、預先回應「n=5 不夠」的質疑。

---

## Key cross-repo links

- **Decision history** (D001–D035): [parent repo `notes/lab_notebook.md`](https://github.com/Hachi282/llm-cooperation-thesis/blob/main/notes/lab_notebook.md)
- **1-page thesis ground truth**: [parent repo `notes/thesis_spine.md`](https://github.com/Hachi282/llm-cooperation-thesis/blob/main/notes/thesis_spine.md)
- **Research question chain v1 → v3**: [parent repo `notes/research_question.md`](https://github.com/Hachi282/llm-cooperation-thesis/blob/main/notes/research_question.md)
- **A002 deviation audit (#7-9 incl. clamp, system-prompt placeholder, inheritance format)**: [parent repo `notes/experiments.md`](https://github.com/Hachi282/llm-cooperation-thesis/blob/main/notes/experiments.md)
- **OpenAI sibling repo**: [Hachi282/llm-cooperation-thesis-openai](https://github.com/Hachi282/llm-cooperation-thesis-openai)
- **Thesis figures** (F1-F4): [openai repo `figures/thesis/`](https://github.com/Hachi282/llm-cooperation-thesis-openai/tree/main/figures/thesis)
