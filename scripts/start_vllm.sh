#!/bin/bash

MODEL="$1"
PORT="$2"
TP="$3"
CUDA_DEVICES="$4"
LOG_FILE="$5"

if [[ "$MODEL" == *gemma-2* ]]; then
  max_model_len=8192
elif [[ "$MODEL" == *phi-4* ]]; then
  max_model_len=16384
else
  max_model_len=80000
fi

mkdir -p logs
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES \
vllm serve "$MODEL" \
  --port "$PORT" \
  --trust-remote-code \
  --enforce-eager \
  --tensor-parallel-size "$TP" \
  --max-model-len "$max_model_len" \
  --gpu-memory-utilization 0.85 \
  > "logs/$LOG_FILE" 2>&1 &

echo $!
