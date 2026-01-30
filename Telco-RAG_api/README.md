source .venv/bin/activate

vllm serve Qwen/Qwen3-32B \
--port 8000 \
--dtype auto \
--max-model-len 12480 \
--seed 42 \
--gpu_memory_utilization 0.5

vllm serve Qwen/Qwen3-Embedding-0.6B \
  --port 8001 \
  --seed 42 \
  --gpu_memory_utilization 0.2

python Telco-RAG/Telco-RAG_api/pipeline_offline.py



pip install -r Telco-RAG/Telco-RAG_api/evaluation_system/requirements.txt
python Telco-RAG/Telco-RAG_api/evaluation_system/RAGAS_QA_Generation.py
python Telco-RAG/Telco-RAG_api/evaluation_system/RAGAS_Evaluation