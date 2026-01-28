from pathlib import Path
import json
import numpy as np
from sentence_transformers import SentenceTransformer

in_path_3gpp = Path("data/3gpp/3gpp_chuncks/3gpp_chunks.json")
in_path_web  = Path("data/web/web_chunks/web_chunks.json")
out_dir = Path("data/db")
out_dir.mkdir(parents=True, exist_ok=True)

chunks = []

# chunks from 3gpp
raw = in_path_3gpp.read_text(encoding="utf-8")
s = raw.lstrip()
for line in raw.splitlines():
    line = line.strip()
    if not line:
        continue
    chunks.append(json.loads(line))
    
# chunks from web?
raw = in_path_web.read_text(encoding="utf-8")
s = raw.lstrip()
for line in raw.splitlines():
    line = line.strip()
    if not line:
        continue
    chunks.append(json.loads(line))

# text가 있는 청크만 사용
filtered = [c for c in chunks if c.get("text")]
texts = [c["text"] for c in filtered]
meta  = [{"element_id": c.get("element_id"), "type": c.get("type")} for c in filtered]

print("N texts =", len(texts))
print("example length =", len(texts[0]))

# ====== 임베딩 ======
model = SentenceTransformer("Qwen/Qwen3-Embedding-8B")
embeddings  = model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
print("embeddings:", embeddings.shape)  # (N, 768)

np.save(out_dir / "embeddings.npy", embeddings)
with (out_dir / "meta.jsonl").open("w", encoding="utf-8") as f:
    for m, t in zip(meta, texts):
        m["text"] = t
        f.write(json.dumps(m, ensure_ascii=False) + "\n")
print("saved:", out_dir)





# Requires transformers>=4.51.0
# Requires sentence-transformers>=2.7.0

from sentence_transformers import SentenceTransformer

# Load the model
model = SentenceTransformer("Qwen/Qwen3-Embedding-8B")

# We recommend enabling flash_attention_2 for better acceleration and memory saving,
# together with setting `padding_side` to "left":
# model = SentenceTransformer(
#     "Qwen/Qwen3-Embedding-8B",
#     model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto"},
#     tokenizer_kwargs={"padding_side": "left"},
# )

# The queries and documents to embed
queries = [
    "What is the capital of China?",
    "Explain gravity",
]
documents = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
]

# Encode the queries and documents. Note that queries benefit from using a prompt
# Here we use the prompt called "query" stored under `model.prompts`, but you can
# also pass your own prompt via the `prompt` argument
query_embeddings = model.encode(queries, prompt_name="query")
document_embeddings = model.encode(documents)

# Compute the (cosine) similarity between the query and document embeddings
similarity = model.similarity(query_embeddings, document_embeddings)
print(similarity)
# tensor([[0.7493, 0.0751],
#         [0.0880, 0.6318]])