# bash mcq_execution.sh

#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

mkdir -p /root/logs



/NAS/inno_aidev/users/hbpark/.venv/bin/vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --port 8000 \
  --dtype auto \
  --max-model-len 16000 \
  --seed 42 \
  --gpu_memory_utilization 0.5 \
  > /root/logs/vllm_8000.log 2>&1 & pid_llm=$!

ok=0
for i in $(seq 1 1200); do
  if curl -sf http://127.0.0.1:8000/v1/models >/dev/null; then ok=1; break; fi
  sleep 1
done
if [ "$ok" -ne 1 ]; then
  echo "[ERROR] vLLM LLM(8000) did not become ready. Check /root/logs/vllm_8000.log" >&2
  exit 1
fi



/NAS/inno_aidev/users/hbpark/.venv/bin/vllm serve Qwen/Qwen3-Embedding-0.6B \
  --port 8001 \
  --seed 42 \
  --gpu_memory_utilization 0.2 \
  > /root/logs/vllm_8001.log 2>&1 & pid_emb=$!

ok=0
for i in $(seq 1 1200); do
  if curl -sf http://127.0.0.1:8001/v1/models >/dev/null; then ok=1; break; fi
  sleep 1
done
if [ "$ok" -ne 1 ]; then
  echo "[ERROR] vLLM Embedding(8001) did not become ready. Check /root/logs/vllm_8001.log" >&2
  exit 1
fi



/NAS/inno_aidev/users/hbpark/.venv/bin/vllm serve Qwen/Qwen3-Reranker-0.6B \
  --port 8002 \
  --runner pooling \
  --seed 42 \
  --gpu_memory_utilization 0.2 \
  > /root/logs/vllm_8002.log 2>&1 & pid_rerank=$!

ok=0
for i in $(seq 1 1200); do
  if curl -sf http://127.0.0.1:8002/v1/models >/dev/null; then ok=1; break; fi
  sleep 1
done
if [ "$ok" -ne 1 ]; then
  echo "[ERROR] vLLM Reranker(8002) did not become ready. Check /root/logs/vllm_8002.log" >&2
  exit 1
fi



cd Telco-RAG/Telco-RAG_api

PYTHONPATH=/NAS/inno_aidev/users/hbpark/Telco-RAG/Telco-RAG_api \
python evaluation_system/mcq_generation.py --dataset "$DATASET_PATH"

kill "$pid_llm" "$pid_emb" "$pid_rerank" 2>/dev/null || true