import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.schema import Document
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context, conditional
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset.docstore import InMemoryDocumentStore

try:
    import ujson as json
except ImportError:
    import json


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
META_PATH = os.path.join(BASE_DIR, "data", "db", "meta.jsonl")
MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
EMBED_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
OUTPUT_DIR = os.path.join(BASE_DIR, "Telco-RAG_api", "evaluation_system")
OUTPUT_BASENAME = "ragas_qa_dataset"


def load_documents(meta_path=META_PATH):
    docs = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries = json.loads(line)
            if isinstance(entries, dict):
                entries = [entries]
            for entry in entries:
                text = entry.get("text", "")
                if not text:
                    continue
                metadata = {}
                if "id" in entry:
                    metadata["id"] = entry["id"]
                if "source" in entry:
                    metadata["source"] = entry["source"]
                docs.append(Document(page_content=text, metadata=metadata))
    return docs


def generate_testset(
    meta_path=META_PATH,
    test_size=100,
    distributions=None,
):
    docs = load_documents(meta_path)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto",
    )

    generator_llm = LangchainLLMWrapper(model, tokenizer)
    critic_llm = LangchainLLMWrapper(model, tokenizer)
    embeddings = LangchainEmbeddingsWrapper(model_name=EMBED_MODEL_NAME)

    docstore = InMemoryDocumentStore.from_documents(docs)

    generator = TestsetGenerator.from_langchain(
        generator_llm,
        critic_llm,
        embeddings,
        docstore=docstore,
    )

    if distributions is None:
        distributions = {
            simple: 0.4,
            reasoning: 0.2,
            multi_context: 0.2,
            conditional: 0.2,
        }

    testset = generator.generate_with_langchain_docs(
        documents=docs,
        test_size=test_size,
        distributions=distributions,
        with_debugging_logs=True,
        raise_exceptions=False,
    )
    return testset


def save_testset(testset, output_dir=OUTPUT_DIR, basename=OUTPUT_BASENAME):
    os.makedirs(output_dir, exist_ok=True)
    test_df = testset.to_pandas()

    csv_path = os.path.join(output_dir, f"{basename}.csv")
    jsonl_path = os.path.join(output_dir, f"{basename}.jsonl")

    test_df.to_csv(csv_path, index=False)
    test_df.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)

    return test_df, csv_path, jsonl_path


if __name__ == "__main__":
    testset = generate_testset()
    test_df, csv_path, jsonl_path = save_testset(testset)
