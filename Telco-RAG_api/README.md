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

vllm serve Qwen/Qwen3-Reranker-0.6B \
  --port 8002 \
  --runner pooling \
  --seed 42 \
  --gpu_memory_utilization 0.2



python Telco-RAG/Telco-RAG_api/pipeline_online.py

export QUERY=''
export OPTIONS='{"option 1:...","option 2:..."}'
export ANSWER=''
python Telco-RAG/Telco-RAG_api/pipeline_online.py --query "$QUERY" --options "$OPTIONS" --answer "$ANSWER"

# XLSX schema + summary generation
python Telco-RAG/Telco-RAG_api/pipeline_online.py \
  --input-file Telco-RAG/Telco-RAG_api/evaluation_system/inputs/20230717143814-R8827-5GSSCVT_SVVONRO_C1_INPROG-3_I-GJ-HLOL-ENB-I001-ALL_N_3500_F20593412_DRM_1SEC.xlsx

# Batch mode for all xlsx in a directory
python Telco-RAG/Telco-RAG_api/pipeline_online.py \
  --input-dir Telco-RAG/Telco-RAG_api/inputs \
  --output-dir Telco-RAG/Telco-RAG_api/outputs


cd Telco-RAG/Telco-RAG_api
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
PYTHONPATH=/NAS/inno_aidev/users/hbpark/Telco-RAG/Telco-RAG_api \
python /NAS/inno_aidev/users/hbpark/Telco-RAG/Telco-RAG_api/evaluation_system/TeleQnA_load.py

cd Telco-RAG/Telco-RAG_api
PYTHONPATH=/NAS/inno_aidev/users/hbpark/Telco-RAG/Telco-RAG_api \
python evaluation_system/mcq_generation.py \
 --dataset /NAS/inno_aidev/users/hbpark/Telco-RAG/Telco-RAG_api/evaluation_system/inputs/MCQ_teleqna.json

cd Telco-RAG/Telco-RAG_api
PYTHONPATH=/NAS/inno_aidev/users/hbpark/Telco-RAG/Telco-RAG_api \
python evaluation_system/mcq_evaluation.py



python Telco-RAG/Telco-RAG_api/evaluation_system/jsonl_to_json.py --input OPENENDED_3gpp_rag_eval_qa_100.jsonl
python Telco-RAG/Telco-RAG_api/evaluation_system/jsonl_to_json.py --input OPENENDED_dm_se_rrc_nas_qa_dataset.jsonl



cd Telco-RAG/Telco-RAG_api
PYTHONPATH=/NAS/inno_aidev/users/hbpark/Telco-RAG/Telco-RAG_api \
python evaluation_system/openended_generation.py \
 --dataset /NAS/inno_aidev/users/hbpark/Telco-RAG/Telco-RAG_api/evaluation_system/inputs/OPENENDED_3gpp_rag_eval_qa_100.json

cd Telco-RAG/Telco-RAG_api
PYTHONPATH=/NAS/inno_aidev/users/hbpark/Telco-RAG/Telco-RAG_api \
python evaluation_system/openended_evaluation.py



python Telco-RAG/Telco-RAG_api/pipeline_online.py \
  --input-file Telco-RAG/Telco-RAG_api/evaluation_system/inputs/20230717143814-R8827-5GSSCVT_SVVONRO_C1_INPROG-3_I-GJ-HLOL-ENB-I001-ALL_N_3500_F20593412_DRM_1SEC.xlsx \
  --output-dir Telco-RAG/Telco-RAG_api/outputs














cd Telco-RAG/Telco-RAG_api
PYTHONPATH=/NAS/inno_aidev/users/hbpark/Telco-RAG/Telco-RAG_api \
python evaluation_system/mcq_evaluation.py \
 --dataset /NAS/inno_aidev/users/hbpark/Telco-RAG/Telco-RAG_api/evaluation_system/inputs/MCQ_gpt.json

cd Telco-RAG/Telco-RAG_api
PYTHONPATH=/NAS/inno_aidev/users/hbpark/Telco-RAG/Telco-RAG_api \
python evaluation_system/mcq_evaluation.py \
 --dataset /NAS/inno_aidev/users/hbpark/Telco-RAG/Telco-RAG_api/evaluation_system/inputs/MCQ_gemini.json

cd Telco-RAG/Telco-RAG_api
PYTHONPATH=/NAS/inno_aidev/users/hbpark/Telco-RAG/Telco-RAG_api \
python evaluation_system/mcq_evaluation.py \
 --dataset /NAS/inno_aidev/users/hbpark/Telco-RAG/Telco-RAG_api/evaluation_system/inputs/MCQ_provided.json

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
