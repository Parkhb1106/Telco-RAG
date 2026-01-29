import json
import logging
import os
import numpy as np
import traceback
from src.LLMs.LLM import embedding

def get_embeddings_OpenAILarge_byapi(text):
    
    response = embedding(text)
    return response.data[0].embedding

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_embeddings(series_docs):
    """Add embeddings to each chunk of documents from pre-saved NumPy files."""
    for doc_key, doc_chunks in series_docs.items():
        try:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            embedding_path = os.path.join(base_dir, "3GPP-Release18", "Embeddings", f"Embeddings{doc_key}.npy")
            embeddings = np.load(embedding_path)
        except FileNotFoundError:
            logging.error(f"Embedding file for {doc_key} not found.")
            continue
        except Exception as e:
            logging.error(f"Failed to load embeddings for {doc_key}: {e}")
            continue
        
        text_list=[]
        for chunk in doc_chunks:
            for single_chunk in chunk:
                text_list.append(single_chunk['text'])
        dex ={}
        for i in range(len(text_list)):
            dex[text_list[i]] = embeddings[i]
        updated_chunks = []
        for chunk in doc_chunks:
            for idx, single_chunk in enumerate(chunk):
                try:
                    chunk[idx]['embedding'] = dex[chunk[idx]['text']]
                    updated_chunks.append(chunk[idx])
                except IndexError:
                    logging.warning(f"Embedding index {idx} out of range for {doc_key}.")
                except Exception as e:
                    logging.error(f"Error processing chunk {idx} for {doc_key}: {e}")
        
        series_docs[doc_key] = updated_chunks
    return series_docs

def get_embeddings_custom():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    embeddings_path = os.path.join(base_dir, "data", "db", "embeddings.npy")
    meta_path = os.path.join(base_dir, "data", "db", "meta.jsonl")

    embeddings = np.load(embeddings_path, mmap_mode="r")
    meta = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item, list) and item:
                item = item[0]
            meta.append(item)

    limit = min(len(meta), embeddings.shape[0])
    embedded_doc = [None] * limit
    for i in range(limit):
        m = meta[i]
        embedded_doc[i] = {
            "text": m.get("text"),
            "source": m.get("source"),
            "embedding": np.asarray(embeddings[i]),
        }

    return [embedded_doc]
