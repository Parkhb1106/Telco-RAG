from sentence_transformers import SentenceTransformer
import numpy as np


def get_embeddings_OpenAILarge_byapi(chunked_doc):
    text_list = []
    for chunk in chunked_doc:
        text_list.append(chunk["text"])
    
    # Use Qwen embedding model
    model = SentenceTransformer("Qwen/Qwen3-Embedding-8B")
    embeddings = model.encode(text_list, batch_size=32, normalize_embeddings=True)
    
    dex = dict()
    for i in range(len(embeddings)):
        dex[text_list[i]] = embeddings[i]
    
    for chunk in chunked_doc:
        chunk['embedding'] = dex[chunk['text']]
        chunk['source'] = 'Online'

    return chunked_doc