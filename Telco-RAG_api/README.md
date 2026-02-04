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

python Telco-RAG/Telco-RAG_api/pipeline_online.py

python Telco-RAG/Telco-RAG_api/pipeline_online.py --query "..." --options '{"option 1:...","option 2:..."}' --answer "..."

export QUERY=''
export OPTIONS='{"option 1:...","option 2:..."}'
export ANSWER=''
python Telco-RAG/Telco-RAG_api/pipeline_online.py --query "$QUERY" --options "$OPTIONS" --answer "$ANSWER"

[ex]
python Telco-RAG/Telco-RAG_api/pipeline_online.py --query "In supporting an MA PDU Session, what does Rel-17 enable in terms of 3GPP access over EPC? [3GPP Release 17]" --options '{"option 1: Direct connection of 3GPP access to 5GC", "option 2: Establishment of user-plane resources over EPC", "option 3: Use of NG-RAN access for all user-plane traffic", "option 4: Exclusive use of a non-3GPP access for user-plane traffic"}' --answer "option 2: Establishment of user-plane resources over EPC"

export QUERY='In supporting an MA PDU Session, what does Rel-17 enable in terms of 3GPP access over EPC? [3GPP Release 17]'
export OPTIONS='{"option 1: Direct connection of 3GPP access to 5GC", "option 2: Establishment of user-plane resources over EPC", "option 3: Use of NG-RAN access for all user-plane traffic", "option 4: Exclusive use of a non-3GPP access for user-plane traffic"}'
export ANSWER='option 2: Establishment of user-plane resources over EPC'
python Telco-RAG/Telco-RAG_api/pipeline_online.py --query "$QUERY" --options "$OPTIONS" --answer "$ANSWER"

cd Telco-RAG/Telco-RAG_api
PYTHONPATH=/NAS/inno_aidev/users/hbpark/Telco-RAG/Telco-RAG_api \
python evaluation_system/mcq_evaluation.py





python Telco-RAG/Telco-RAG_api/evaluation_system/RAGAS_QA_Generation.py \
  --embeddings Telco-RAG/data/db/embeddings.npy \
  --meta Telco-RAG/data/db/meta.jsonl \
  --out Telco-RAG/data/evalsets/ragas_evalset.jsonl \
  --testset_size 10

python Telco-RAG/Telco-RAG_api/evaluation_system/RAGAS_Evaluation

cd Telco-RAG/Telco-RAG_api
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
# (paste your Hugging Face token here; do not commit real tokens)

PYTHONPATH=/NAS/inno_aidev/users/hbpark/Telco-RAG/Telco-RAG_api \
python /NAS/inno_aidev/users/hbpark/Telco-RAG/Telco-RAG_api/evaluation_system/mcp_TeleQnA.py --limit 10 --shuffle

PYTHONPATH=/NAS/inno_aidev/users/hbpark/Telco-RAG/Telco-RAG_api \
python /NAS/inno_aidev/users/hbpark/Telco-RAG/Telco-RAG_api/evaluation_system/mcp_TeleQnA.py --sleep 0.2
