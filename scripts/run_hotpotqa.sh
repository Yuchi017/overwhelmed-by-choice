#!/bin/bash
# Reproduce the HotpotQA experiments for one model: single-pass, CoT, hierarchical
# partitioning, permutation inference, the length-controlled and irrelevant-context
# variants, and the gold-margin analysis.
#
# Usage: bash scripts/run_hotpotqa.sh [MODEL] [GPU_IDS] [TENSOR_PARALLEL]
#   e.g. bash scripts/run_hotpotqa.sh meta-llama/Llama-3.1-8B-Instruct 0 1
set -e
cd "$(dirname "$0")/.."

MODEL="${1:-meta-llama/Llama-3.1-8B-Instruct}"
GPUS="${2:-0}"
TP="${3:-1}"
PORT="${PORT:-8000}"
TAG="${MODEL##*/}"
RUN="python scripts/run_inference.py --model $MODEL --port $PORT --dataset beir --max-samples 200"

./scripts/start_vllm.sh "$MODEL" "$PORT" "$TP" "$GPUS" "vllm_${TAG}.log" > /dev/null
trap 'lsof -ti tcp:${PORT} | xargs -r kill' EXIT
./scripts/wait_vllm_ready.sh "$PORT"

# Main results (N = 5 ... 160)
$RUN --method single       --data-path data/hotpotqa
$RUN --method single       --data-path data/hotpotqa --enable-cot
$RUN --method hierarchical --data-path data/hotpotqa --group-size 5
$RUN --method permutation  --data-path data/hotpotqa --num-shuffles 5

# Distractor verbosity: short vs long passages
$RUN --method single --data-path data/hotpotqa_short --max-tokens 200
$RUN --method single --data-path data/hotpotqa_long  --max-tokens 200

# Irrelevant context: prepend k copies of data/noise.txt (~47 tokens each) at N = 40
for K in 0 5 10 20 40 80 160; do
  $RUN --method single --data-path data/hotpotqa --tag hotpotqa_noise --num-options 40 --noise-repeats $K
done

# Stop vLLM before loading the model with transformers for the log-prob analysis
lsof -ti tcp:${PORT} | xargs -r kill; sleep 10
export CUDA_VISIBLE_DEVICES="${GPUS%%,*}"
python scripts/score_options.py --source single       --model "$MODEL"
python scripts/score_options.py --source single       --model "$MODEL" --enable-cot
python scripts/score_options.py --source hierarchical --model "$MODEL"
python scripts/score_options.py --source permutation  --model "$MODEL"
