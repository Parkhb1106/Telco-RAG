import os
import time
import argparse
import traceback
from typing import List, Dict, Any, Optional

import pandas as pd
import torch
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision, answer_similarity
#from ragas.llms import LangchainLLMWrapper
#from ragas.embeddings import LangchainEmbeddingsWrapper
#from transformers import AutoModelForCausalLM, AutoTokenizer

from pipeline_offline import TelcoRAG

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DATASET_DIR = os.path.join(BASE_DIR, "evaluation_system")
DEFAULT_BASENAME = "ragas_qa_dataset"

DEFAULT_MODEL_NAME = os.getenv("TELCORAG_MODEL_NAME", "Qwen/Qwen3-32B")
EVAL_LLM_MODEL = os.getenv("RAGAS_EVAL_LLM", "Qwen/Qwen3-30B-A3B-Instruct-2507")
EVAL_EMBED_MODEL = os.getenv("RAGAS_EVAL_EMBED", "Qwen/Qwen3-Embedding-0.6B")


def _resolve_dataset_path(dataset_dir: str, basename: str) -> str:
    jsonl_path = os.path.join(dataset_dir, f"{basename}.jsonl")
    csv_path = os.path.join(dataset_dir, f"{basename}.csv")
    if os.path.isfile(jsonl_path):
        return jsonl_path
    if os.path.isfile(csv_path):
        return csv_path
    raise FileNotFoundError(f"Dataset not found: {jsonl_path} or {csv_path}")


def load_ragas_dataset() -> pd.DataFrame:
    path = _resolve_dataset_path(DEFAULT_DATASET_DIR, DEFAULT_BASENAME)
    if path.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    return pd.read_csv(path)



def build_eval_dataset(test_df: pd.DataFrame) -> Dataset:
    rows = test_df.iloc[0:]
    outputs = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    gt_key = "ground_truth" if "ground_truth" in rows.columns else ("answer" if "answer" in rows.columns else None)

    for _, row in rows.iterrows():
        question = str(row["question"])
        answer, contexts = TelcoRAG(query=question, answer=None, options=None, model_name=DEFAULT_MODEL_NAME)
        outputs["user_input"].append(question)
        outputs["response"].append(answer)
        outputs["retrieved_contexts"].append(contexts)
        outputs["reference"].append("" if gt_key is None else str(row[gt_key]))
    return Dataset.from_dict(outputs)


def main(): 
    parser = argparse.ArgumentParser(description="Evaluate Telco-RAG using RAGAS framework.")
    parser.add_argument("--output", type=str, default="ragas_evaluation_results.csv", help="Path to save")
    args = parser.parse_args()

    try:
        test_df = load_ragas_dataset()
        eval_dataset = build_eval_dataset(test_df)
        
        #tokenizer = AutoTokenizer.from_pretrained(EVAL_LLM_MODEL)
        #model = AutoModelForCausalLM.from_pretrained(
        #    EVAL_LLM_MODEL,
        #    torch_dtype="auto",
        #    device_map="auto",
        #)
        #llm = LangchainLLMWrapper(model, tokenizer)
        #embeddings = LangchainEmbeddingsWrapper(model_name=EVAL_EMBED_MODEL)

        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
            #answer_similarity,
        ]
        #result = evaluate(eval_dataset, metrics=metrics, llm=llm, embeddings=embeddings)
        result = evaluate(eval_dataset, metrics=metrics)

        result_df = result.to_pandas()
        result_df.to_csv(args.output, index=False)
        print(result_df)
        print(f"Saved results to: {args.output}")
    except Exception as e:
        print(f"Evaluation failed: {e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
