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
    
# chunks from web
chunks.extend(json.loads(in_path_web.read_text(encoding="utf-8")))

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
