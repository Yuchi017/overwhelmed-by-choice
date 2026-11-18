#!/bin/bash
# Reproduce the HotpotQA results: single-pass, hierarchical partitioning, and permutation
# inference across candidate-set sizes N. Starts a vLLM server, runs all methods, then stops it.
#
# Usage: bash scripts/run_hotpotqa.sh [MODEL] [GPU_IDS] [TP]
#   e.g. bash scripts/run_hotpotqa.sh meta-llama/Llama-3.1-8B-Instruct 0 1
set -e
cd "$(dirname "$0")/.."

MODEL="${1:-meta-llama/Llama-3.1-8B-Instruct}"
GPUS="${2:-0}"
TP="${3:-1}"
PORT="${PORT:-8000}"
DATASET_NAME=hotpotqa
DATASET_PATH=data/hotpotqa
MAX_SAMPLES=200
NUM_OPTIONS_LIST=(5 10 20 40 80 160)
GROUP_SIZE=5
NUM_SHUFFLES=5
TAG="${MODEL##*/}"

./scripts/start_vllm.sh "$MODEL" "$PORT" "$TP" "$GPUS" "vllm_${TAG}.log" > /dev/null
trap 'lsof -ti tcp:${PORT} | xargs -r kill' EXIT
./scripts/wait_vllm_ready.sh "$PORT"

COMMON=(--vllm_port "$PORT" --model "$MODEL" --dataset-class Bier --dataset-path "$DATASET_PATH" --max-samples "$MAX_SAMPLES")

for N in "${NUM_OPTIONS_LIST[@]}"; do
  # 1) Single-pass baseline (add --enable-cot for the CoT variant)
  out="results/single/${TAG}/${DATASET_NAME}/option${N}_sample${MAX_SAMPLES}.json"
  [[ -f $out ]] || python eval.py "${COMMON[@]}" --num-options "$N" --output "$out"

  # 2) Hierarchical partitioning
  out="results/group/${TAG}/${DATASET_NAME}/option${N}_sample${MAX_SAMPLES}_group${GROUP_SIZE}.json"
  [[ -f $out ]] || python eval_group.py "${COMMON[@]}" --num-options "$N" --group "$GROUP_SIZE" --output "$out"

  # 3) Permutation-based inference
  out="results/shuffle/${TAG}/${DATASET_NAME}/option${N}_sample${MAX_SAMPLES}_shuffle${NUM_SHUFFLES}.json"
  [[ -f $out ]] || python eval_shuffle.py "${COMMON[@]}" --num-options "$N" --num-shuffles "$NUM_SHUFFLES" --shuffle-seed-base 42 --output "$out"
done
