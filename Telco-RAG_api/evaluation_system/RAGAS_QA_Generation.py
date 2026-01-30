import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_core.documents import Document
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import default_query_distribution
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

try:
    import ujson as json
except ImportError:
    import json
    
try:
    from langchain_community.llms import HuggingFacePipeline
except ImportError:
    from langchain_huggingface import HuggingFacePipeline  # 일부 환경

try:
    from langchain_community.embeddings import HuggingFaceEmbeddings as LCHuggingFaceEmbeddings
except ImportError:
    from langchain_huggingface import HuggingFaceEmbeddings as LCHuggingFaceEmbeddings  # 일부 환경



BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
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
    test_size=40,
):
    docs = load_documents(meta_path)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto",
    )

    gen_pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=False,       # 테스트셋 생성 안정성 위해 보통 False 권장
        temperature=0.0,
        return_full_text=False,
    )
    lc_llm = HuggingFacePipeline(pipeline=gen_pipe)
    generator_llm = LangchainLLMWrapper(lc_llm)
    
    lc_emb = LCHuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    generator_embeddings = LangchainEmbeddingsWrapper(lc_emb)

    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings,
    )

    query_distribution = default_query_distribution(llm=generator_llm)

    testset = generator.generate_with_chunks(
        chunks=docs,
        testset_size=test_size,
        query_distribution=query_distribution,
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
    print("Saved:", csv_path, jsonl_path)
