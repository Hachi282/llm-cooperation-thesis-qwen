#!/usr/bin/env bash
# D032 ablation: 5 seeds with --inherit-format scored.
# Everything else identical to run_5_seeds.sh (the Qwen v2 baseline).
set -e
cd "$(dirname "$0")"
mkdir -p logs

for SEED in 7 13 42 99 2024; do
    TAG="qwen_v2_scored_s${SEED}"
    LOG="logs/${TAG}_run.log"
    if [ -f "logs/${TAG}.jsonl" ] && grep -q '"type": "experiment_done"' "logs/${TAG}.jsonl"; then
        echo "############ ${TAG} already complete, skipping ############"
        continue
    fi
    rm -f "logs/${TAG}.jsonl"
    echo ""
    echo "############ Starting ${TAG} ($(date +%H:%M:%S)) ############"
    python -X utf8 main.py --seed "$SEED" --tag "$TAG" --inherit-format scored > "$LOG" 2>&1
    echo "############ Finished ${TAG} ($(date +%H:%M:%S)) ############"
done

echo ""
echo "ALL 5 SCORED-ABLATION SEEDS DONE at $(date +%H:%M:%S)"
