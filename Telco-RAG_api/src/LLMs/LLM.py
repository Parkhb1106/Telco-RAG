import openai
import tiktoken
from sentence_transformers import SentenceTransformer
from functools import lru_cache
import asyncio
from typing import Any, Dict, List, Optional, Tuple
import requests
import math

import anthropic # type: ignore
from mistralai.async_client import MistralAsyncClient
from mistralai.client import MistralClient

from together import AsyncTogether, Together


import time

from src.LLMs.settings.config import get_settings
from groq import Groq, AsyncGroq

import platform
if platform.system()=='Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
settings = get_settings()
rate_limit = settings.rate_limit 

# API keys
openai.api_key = settings.openai_api_key
any_api_key = settings.any_api_key
mistral_api = settings.mistral_api
anthropic_api = settings.anthropic_api
cohere_api = settings.cohere_api
pplx_api = settings.pplx_api
together_api = settings.together_api

groq_api = ""

# Models config
models = [
    "gpt-4o-mini",
    "gpt-4",
    'mixtral',
    'mistral-small',
    'mistral-medium',
    "code-llama",
    "command-R+",
    'pplx',
    'mixtral-8x22',
    "mixtral-groq",
    'llama-3',
    'llama-3-any',
    'llama-3-8B',
    'wizard'
]

models_fullnames = {
    "gpt-3.5": "gpt-4o-mini",
    "gpt-4": "gpt-4-turbo-2024-04-09",
    "gpt-4o": "gpt-4o-2024-05-13",
    "gpt-4o-mini": "gpt-4o-mini",
    "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistral-small-old": 'open-mixtral-8x7b',
    "mistral-small": 'mistral-small-latest',
    "mistral-medium": 'mistral-medium-latest',
    "mistral-large": 'mistral-large-latest',
    "code-llama": "codellama/CodeLlama-70b-Instruct-hf",
    "claude-small": "claude-3-haiku-20240307",
    "claude-medium": "claude-3-sonnet-20240229",
    "claude-large": "claude-3-opus-20240229",
    "command-R+": "command-r-plus",
    "pplx": "llama-3-sonar-large-32k-online",
    'mixtral-8x22': "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "mixtral-groq": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    'llama-3': 'meta-llama/Llama-3-70b-chat-hf',
    'llama-3-any': 'meta-llama/Meta-Llama-3-70B-Instruct',
    'llama-3-8B' : 'meta-llama/Llama-3-8b-chat-hf',
    'wizard': "microsoft/WizardLM-2-8x22B"
}

models_endpoints = {
    "gpt-3.5": "openai",
    "gpt-4": "openai",
    "gpt-4o": "openai",
    "gpt-4o-mini": "openai",
    "mixtral": "anyscale",
    "mistral-small-old": 'mistral',
    "mistral-small": 'mistral',
    "mistral-medium": 'mistral',
    "mistral-large": 'mistral',
    "code-llama": "anyscale",
    "claude-small": "anthropic",
    "claude-medium": "anthropic",
    "claude-large": "anthropic",
    "command-R+": "cohere",
    "pplx": "perplexity",
    'mixtral-8x22': 'anyscale',
    "mixtral-groq": 'groq',
    'llama-3': 'together',
    'llama-3-any': 'anyscale',
    'llama-3-8B': 'anyscale',
    'wizard': 'together'
}

token_prices = {
    "gpt-3.5": 0.0015 / 1000,
    "gpt-4": 0.06 / 1000,
    "mixtral": 0.5 / 1000000,
    "mixtral-groq": 0.5 / 1000000,
    "mistral-medium": 5 / 1000000,
}


class RateLimiter:
    def __init__(self, calls_per_second=0.25):
        self.calls_per_second = rate_limit
        self.semaphore = asyncio.Semaphore(calls_per_second)
        self.next_call_time = time.time()

    async def wait_for_rate_limit(self):
        async with self.semaphore:
            now = time.time()
            sleep_time = self.next_call_time - now
            self.next_call_time = max(self.next_call_time + 1 / self.calls_per_second, now)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            self.next_call_time = max(self.next_call_time + 1 / self.calls_per_second, now)


def submit_prompt_flex(prompt, model="gpt-4o-mini", output_json=False):
    if model in models_fullnames:
        model_fullname = models_fullnames[model]
        endpoint = models_endpoints[model]
    else:
        endpoint = ""
        
    if endpoint == "anyscale":
        print(f"Endpoint: {endpoint}")
        print(f"Model: {model_fullname}")
        client = openai.OpenAI(
            base_url = "https://api.endpoints.anyscale.com/v1",
            api_key=any_api_key,
        )
        generate = client.chat.completions.create
    elif endpoint == "perplexity":
        print(f"Endpoint: {endpoint}")
        print(f"Model: {model_fullname}")
        client = openai.OpenAI(
            base_url = "https://api.perplexity.ai",
            api_key=any_api_key,
        )
        generate = client.chat.completions.create
    elif endpoint == "groq":
        print(f"Endpoint: {endpoint}")
        print(f"Model: {model_fullname}")
        client = Groq(
            api_key=groq_api,
        )
        generate = client.chat.completions.create
    elif endpoint == "together":
        print(f"Endpoint: {endpoint}")
        print(f"Model: {model_fullname}")     
        client = Together(api_key=together_api)  
        generate = client.chat.completions.create 
    elif endpoint == "openai":
        print(f"Endpoint: {endpoint}")
        print(f"Model: {model_fullname}")
        client = openai.OpenAI(
            api_key=openai.api_key,
        )
        generate = client.chat.completions.create
    elif endpoint == "mistral":
        print(f"Endpoint: {endpoint}")
        print(f"Model: {model_fullname}")
        client = MistralClient(api_key=mistral_api)
        generate = client.chat
    elif endpoint == "anthropic":
        print(f"Endpoint: {endpoint}")
        print(f"Model: {model_fullname}")
        client = anthropic.Anthropic(
            api_key=anthropic_api,
        )
        def generate(**kwargs):
            return client.messages.create(
                max_tokens=4000,
                **kwargs
            )
    else:
        model_fullname = model
        print(f"Endpoint: {endpoint}")
        print(f"Model: {model_fullname}")
        client = openai.OpenAI(
            base_url = "http://localhost:8000/v1",
            api_key=any_api_key,
        )
        generate = client.chat.completions.create        

    if output_json:
        generated_output = generate(
          model=model_fullname,
          response_format={"type":"json_object"},
          messages=[
              {"role": "user", "content": prompt}, 
            ]
        )
        if endpoint != "anthropic":
            output = generated_output.choices[0].message.content
        else:
            output = generated_output.content[0].text
            
        output = output.replace('"\n', '",\n')
        output = output[:output.rfind("}")+1]
        
    else:
        generated_output = generate(
          model=model_fullname,
          messages=[
              {"role": "user", "content": prompt}, 
            ]
        )
        if endpoint != "anthropic":
            output = generated_output.choices[0].message.content
        else:
            output = generated_output.content[0].text
    
    return output


async def a_submit_prompt_flex(prompt, model="gpt-4o-mini", output_json=False):
    if model in models_fullnames:
        model_fullname = models_fullnames[model]
        endpoint = models_endpoints[model]
    else:
        endpoint = ""
        
    if endpoint == "anyscale":
        print(f"Endpoint: {endpoint}")
        print(f"Model: {model_fullname}")
        client = openai.AsyncOpenAI(
            base_url = "https://api.endpoints.anyscale.com/v1",
            api_key=any_api_key,
        )
        generate = client.chat.completions.create
    elif endpoint == "perplexity":
        print(f"Endpoint: {endpoint}")
        print(f"Model: {model_fullname}")
        client = openai.AsyncOpenAI(
            base_url = "https://api.perplexity.ai",
            api_key=any_api_key,
        )
        generate = client.chat.completions.create
    elif endpoint == "groq":
        print(f"Endpoint: {endpoint}")
        print(f"Model: {model_fullname}")
        client = AsyncGroq(
            api_key=groq_api,
        )
        generate = client.chat.completions.create
    elif endpoint == "together":
        print(f"Endpoint: {endpoint}")
        print(f"Model: {model_fullname}")     
        client = AsyncTogether(api_key=together_api)  
        generate = client.chat.completions.create 
    elif endpoint == "openai":
        print(f"Endpoint: {endpoint}")
        print(f"Model: {model_fullname}")
        client = openai.AsyncOpenAI(
            api_key=openai.api_key,
        )
        generate = client.chat.completions.create
    elif endpoint == "mistral":
        print(f"Endpoint: {endpoint}")
        print(f"Model: {model_fullname}")
        client = MistralAsyncClient(api_key=mistral_api)
        generate = client.chat
    elif endpoint == "anthropic":
        print(f"Endpoint: {endpoint}")
        print(f"Model: {model_fullname}")
        client = anthropic.AsyncAnthropic(
            api_key=anthropic_api,
        )
        async def generate(**kwargs):
            return await client.messages.create(
                max_tokens=4000,
                **kwargs
            )
    else:
        model_fullname = model
        print(f"Endpoint: {endpoint}")
        print(f"Model: {model_fullname}")
        client = openai.AsyncOpenAI(
            base_url = "http://localhost:8000/v1",
            api_key=any_api_key,
        )
        generate = client.chat.completions.create       

    if output_json:
        generated_output = await generate(
          model=model_fullname,
          response_format={"type":"json_object"},
          messages=[
              {"role": "user", "content": prompt}, 
            ]
        )
        if endpoint != "anthropic":
            output = generated_output.choices[0].message.content
        else:
            output = generated_output.content[0].text
            
        output = output.replace('"\n', '",\n')
        output = output[:output.rfind("}")+1]
        
    else:
        generated_output = await generate(
          model=model_fullname,
          messages=[
              {"role": "user", "content": prompt}, 
            ]
        )
        if endpoint != "anthropic":
            output = generated_output.choices[0].message.content
        else:
            output = generated_output.content[0].text
    
    return output

def embedding(input, dimension=1024):
    client = openai.OpenAI(api_key=openai.api_key, base_url="http://localhost:8001/v1/")
    response = client.embeddings.create(
                    input=input,
                    # NNRouter는 1024-d 임베딩에 맞춰져 있으므로 1024-d 모델 사용
                    model="Qwen/Qwen3-Embedding-0.6B",
                )
    return response

    # class _EmbeddingItem:
    #     def __init__(self, embedding):
    #         self.embedding = embedding

    # class _EmbeddingResponse:
    #     def __init__(self, embeddings):
    #         self.data = [_EmbeddingItem(emb) for emb in embeddings]

    # @lru_cache(maxsize=1)
    # def _get_embedder():
    #     return SentenceTransformer("BAAI/bge-large-en-v1.5")

    # model = _get_embedder()
    # vecs = model.encode(input, normalize_embeddings=True)
    # embeddings = vecs.tolist() if hasattr(vecs, "tolist") else list(vecs)
    # if embeddings and isinstance(embeddings[0], (int, float)):
    #     embeddings = [embeddings]
    # return _EmbeddingResponse(embeddings)

@lru_cache(maxsize=2)
def _get_vllm_reranker_components(model_name: str):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
    true_token = tokenizer("yes", add_special_tokens=False).input_ids[0]
    false_token = tokenizer("no", add_special_tokens=False).input_ids[0]
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1,
        logprobs=20,
        allowed_token_ids=[true_token, false_token],
    )
    llm = LLM(
        model=model_name,
        tensor_parallel_size=max(1, torch.cuda.device_count()),
        max_model_len=10000,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.8,
    )
    return llm, tokenizer, sampling_params, suffix_tokens, true_token, false_token

def rerank_vllm(
    query: str,
    documents: List[str],
    top_n: Optional[int] = None,
    *,
    base_url: str = "http://localhost:8002/v1",
    model: str = "Qwen/Qwen3-Reranker-0.6B",
    timeout: float = 60.0,
) -> Dict[str, Any]:
    """
    vLLM 로컬 엔진 기반 rerank.
    반환 포맷은 기존 /rerank endpoint 응답과 동일하게 유지:
    {'results': [{'index', 'relevance_score', 'document': {'text'}}]}
    """
    if not documents:
        return {"model": model, "query": query, "results": []}

    from vllm.inputs.data import TokensPrompt
    llm, tokenizer, sampling_params, suffix_tokens, true_token, false_token = _get_vllm_reranker_components(model)

    instruction = "Given a web search query, retrieve relevant passages that answer the query"
    pairs = [(query, doc) for doc in documents]
    messages = [
        [
            {
                "role": "system",
                "content": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".",
            },
            {
                "role": "user",
                "content": f"<Instruct>: {instruction}\n\n<Query>: {q}\n\n<Document>: {d}",
            },
        ]
        for q, d in pairs
    ]
    try:
        tokenized_messages = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False, enable_thinking=False
        )
    except TypeError:
        tokenized_messages = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        )

    max_length = 8192
    prompt_token_ids = [ids[: max_length - len(suffix_tokens)] + suffix_tokens for ids in tokenized_messages]
    prompts = [TokensPrompt(prompt_token_ids=ids) for ids in prompt_token_ids]
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

    scored_results: List[Dict[str, Any]] = []
    for idx, out in enumerate(outputs):
        final_logits = out.outputs[0].logprobs[-1] if out.outputs and out.outputs[0].logprobs else {}
        true_logit = final_logits[true_token].logprob if true_token in final_logits else -10.0
        false_logit = final_logits[false_token].logprob if false_token in final_logits else -10.0
        true_score = math.exp(true_logit)
        false_score = math.exp(false_logit)
        relevance_score = true_score / (true_score + false_score)
        scored_results.append(
            {
                "index": idx,
                "relevance_score": float(relevance_score),
                "document": {"text": documents[idx]},
            }
        )

    scored_results.sort(key=lambda x: x["relevance_score"], reverse=True)
    if top_n is not None:
        scored_results = scored_results[:top_n]

    return {"model": model, "query": query, "results": scored_results}

def rerank_passages(
    query: str,
    passages: List[str],
    top_n: Optional[int] = None,
    *,
    base_url: str = "http://localhost:8002/v1",
    model: str = "Qwen/Qwen3-Reranker-0.6B",
) -> List[Tuple[int, float, str]]:
    """
    편하게 쓰라고 만든 래퍼:
    반환: [(original_index, relevance_score, passage_text), ...]  (score 내림차순)
    """
    out = rerank_vllm(
        query=query,
        documents=passages,
        top_n=top_n,
        base_url=base_url,
        model=model,
    )
    results = out.get("results", [])
    ranked: List[Tuple[int, float, str]] = []
    for r in results:
        idx = int(r["index"])
        score = float(r.get("relevance_score", 0.0))
        # vLLM 예시 응답에서는 document: {text: "..."} 형태 :contentReference[oaicite:3]{index=3}
        doc_text = r.get("document", {}).get("text", passages[idx] if 0 <= idx < len(passages) else "")
        ranked.append((idx, score, doc_text))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked
