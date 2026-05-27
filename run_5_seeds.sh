#!/usr/bin/env bash
# Sequentially run all 5 seeds (7/13/42/99/2024) at T=0.8 via Ollama.
# Ollama serializes 7B inference internally; parallel would just queue.
# Expected: ~45-60 min per seed → ~4-5 hr total.
set -e
cd "$(dirname "$0")"
mkdir -p logs

for SEED in 7 13 42 99 2024; do
    TAG="qwen_v2_s${SEED}"
    LOG="logs/${TAG}_run.log"
    if [ -f "logs/${TAG}.jsonl" ] && grep -q '"type": "experiment_done"' "logs/${TAG}.jsonl"; then
        echo "############ ${TAG} already complete, skipping ############"
        continue
    fi
    # Wipe partial log so each run starts clean.
    rm -f "logs/${TAG}.jsonl"
    echo ""
    echo "############ Starting ${TAG} ($(date +%H:%M:%S)) ############"
    python -X utf8 main.py --seed "$SEED" --tag "$TAG" > "$LOG" 2>&1
    echo "############ Finished ${TAG} ($(date +%H:%M:%S)) ############"
done

echo ""
echo "ALL 5 SEEDS DONE at $(date +%H:%M:%S)"
