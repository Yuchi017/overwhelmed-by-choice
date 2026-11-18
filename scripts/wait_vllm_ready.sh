#!/bin/bash
PORT="$1"
TIMEOUT=1000 # seconds
INTERVAL=5

elapsed=0
while true; do
  if curl -s http://localhost:$PORT/v1/models > /dev/null; then
    echo "vLLM ready on port $PORT"
    break
  fi

  sleep $INTERVAL
  elapsed=$((elapsed + INTERVAL))

  if [ $elapsed -ge $TIMEOUT ]; then
    echo "Timeout waiting for vLLM"
    exit 1
  fi
done
