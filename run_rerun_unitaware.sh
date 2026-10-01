#!/usr/bin/env bash
# AAMAS E0 follow-up (2026-10-01): rerun Qwen2.5-7B v2 baseline (list) and the
# D032 scored ablation with --unit-aware-parse. The original runs misread 481
# answers like "50% of 27.4 = 13.7" as 50 units, which changed the game
# trajectory; the advisor's spec requires affected conditions to be rerun.
# Same seeds, model, temperature and code path; only the parser differs.
set -e
cd "$(dirname "$0")"
mkdir -p logs
for FMT in list scored; do
  for SEED in 7 13 42 99 2024; do
    if [ "$FMT" = list ]; then TAG="qwen_v2u_s${SEED}"; else TAG="qwen_v2u_scored_s${SEED}"; fi
    if [ -f "logs/${TAG}.jsonl" ] && grep -q '"type": "experiment_done"' "logs/${TAG}.jsonl"; then
      echo "### ${TAG} already complete, skipping"; continue
    fi
    rm -f "logs/${TAG}.jsonl"
    echo "### Starting ${TAG} ($(date +%H:%M:%S))"
    python -X utf8 main.py --seed "$SEED" --tag "$TAG" --inherit-format "$FMT" --unit-aware-parse > "logs/${TAG}_run.log" 2>&1
    echo "### Finished ${TAG} ($(date +%H:%M:%S))"
  done
done
echo "ALL 10 RERUNS DONE at $(date +%H:%M:%S)"
