source .venv/bin/activate

vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 \
--port 8000 \
--dtype auto \
--max-model-len 16000 \
--seed 42 \
--gpu_memory_utilization 0.5

vllm serve Qwen/Qwen3-Embedding-0.6B \
  --port 8001 \
  --seed 42 \
  --gpu_memory_utilization 0.2

python Telco-RAG/Telco-RAG_api/pipeline_offline.py
python Telco-RAG/Telco-RAG_api/pipeline_online.py




python Telco-RAG/Telco-RAG_api/evaluation_system/RAGAS_QA_Generation.py \
  --embeddings Telco-RAG/data/db/embeddings.npy \
  --meta Telco-RAG/data/db/meta.jsonl \
  --out Telco-RAG/data/evalsets/ragas_evalset.jsonl \
  --testset_size 10

python Telco-RAG/Telco-RAG_api/evaluation_system/RAGAS_Evaluation

cd Telco-RAG/Telco-RAG_api
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
# [REDACTED_HF_TOKEN]

PYTHONPATH=/NAS/inno_aidev/users/hbpark/Telco-RAG/Telco-RAG_api \
python /NAS/inno_aidev/users/hbpark/Telco-RAG/Telco-RAG_api/evaluation_system/mcp_TeleQnA.py --limit 10 --shuffle

PYTHONPATH=/NAS/inno_aidev/users/hbpark/Telco-RAG/Telco-RAG_api \
python /NAS/inno_aidev/users/hbpark/Telco-RAG/Telco-RAG_api/evaluation_system/mcp_TeleQnA.py --sleep 0.2

