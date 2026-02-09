# bash mcq_evaluation.sh

#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate



cd Telco-RAG/Telco-RAG_api

PYTHONPATH=/NAS/inno_aidev/users/hbpark/Telco-RAG/Telco-RAG_api \
python evaluation_system/mcq_evaluation.py

ACCURACY=$(python -c 'import json; print(json.load(open("evaluation_system/outputs/mcq/mcq_summary.json"))["accuracy"])')

python -c 'import os, vessl; vessl.update_context_variables(data={"ACCURACY": os.environ.get("ACCURACY","")})'