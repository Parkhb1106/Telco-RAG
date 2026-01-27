python3 -m venv .venv
source .venv/bin/activate
pip install unstructured python-docx
python data/3gpp/clean_docx.py
python data/3gpp/chunking.py
