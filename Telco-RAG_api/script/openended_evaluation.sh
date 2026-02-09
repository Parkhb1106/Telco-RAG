# bash mcq_evaluation.sh

#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate



cd Telco-RAG/Telco-RAG_api

PYTHONPATH=/NAS/inno_aidev/users/hbpark/Telco-RAG/Telco-RAG_api \
python evaluation_system/openended_evaluation.py --llm "$LLM"

SIMILARITY=$(python -c 'import json; print(json.load(open("evaluation_system/outputs/open_ended/evaluation_summary.json"))["similarity"])')
BERT_SCORE=$(python -c 'import json; print(json.load(open("evaluation_system/outputs/open_ended/evaluation_summary.json"))["bert_score"])')
FAITHFULNESS=$(python -c 'import json; print(json.load(open("evaluation_system/outputs/open_ended/evaluation_summary.json"))["faithfulness"])')

python -c 'import os, vessl; vessl.update_context_variables(data={"ACCURACY": os.environ.get("ACCURACY","")}); vessl.update_context_variables(data={"BERT_SCORE": os.environ.get("BERT_SCORE","")}); vessl.update_context_variables(data={"FAITHFULNESS": os.environ.get("FAITHFULNESS","")})'