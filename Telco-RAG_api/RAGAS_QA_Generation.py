# TODO: RAGAS_QA_Generation.py 파일에서 데이터셋 생성 코드 작성
from langchain_community.document_loaders import PDFPlumberLoader
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context,
conditional
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset.docstore import InMemoryDocumentStore


# 데이터셋 생성기
tokenizer = AutoTokenizer.from_pretrained(model_name="Qwen/Qwen3-30B-A3B-Instruct-2507")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
generator_llm = LangchainLLMWrapper(model, tokenizer)
critic_llm = LangchainLLMWrapper(model, tokenizer)
embeddings = LangchainEmbeddingsWrapper(model_name="Qwen/Qwen3-Embedding-0.6B")

docstore = InMemoryDocumentStore.from_documents("../data/db/meta.jsonl")

generator = TestsetGenerator.from_langchain(
    generator_llm,
    critic_llm,
    embeddings,
    docstore=docstore,
)

# 질문 유형별 분포 설정
distributions = {
    simple: 0.4,        # 단순 질의
    reasoning: 0.2,     # 추론이 필요한 질의
    multi_context: 0.2,# 여러 문서 맥락이 필요한 질의
    conditional: 0.2    # 조건부 질의
}

# 테스트셋 생성 (100개 샘플)
testset = generator.generate_with_langchain_docs(
    documents=docs,
    test_size=10,
    distributions=distributions,
    with_debugging_logs=True,
    raise_exceptions=False,
)

# Pandas DataFrame으로 변환
test_df = testset.to_pandas()
