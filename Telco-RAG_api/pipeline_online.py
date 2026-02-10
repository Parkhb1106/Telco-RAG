import os
from pathlib import Path
import traceback
from src.query import Query
from src.generate import generate, check_question, analyze_xlsx
from src.LLMs.LLM import submit_prompt_flex
from src.xlsx_schema import extract_xlsx_preview, _build_rag_query
import git
import asyncio
import time
try:
    import ujson as jsonlib
except ImportError:
    import json as jsonlib

folder_url = "https://huggingface.co/datasets/netop/Embeddings3GPP-R18"
clone_directory = "./3GPP-Release18"

if not os.path.exists(clone_directory):
    git.Repo.clone_from(folder_url, clone_directory)
    print("Folder cloned successfully!")
else:
    print("Folder already exists. Skipping cloning.")

async def TelcoRAG(
    query,
    answer=None,
    options=None,
    model_name='gpt-4o-mini',
    xlsx_file: str | None = None,
    max_sample_rows: int = 2000,
    max_scan_rows: int = 2000,
):
    try:
        start =  time.time()
        if xlsx_file:
            preview = extract_xlsx_preview(
                xlsx_path=xlsx_path,
                max_sample_rows=max_sample_rows,
                max_scan_rows=max_scan_rows,
            )
            
            question_text = _build_rag_query(preview)
            question = Query(question_text, [])
            
            question.def_TA_question(isxlsx = True)
            
            semantic_search = question.get_custom_context(k=10, model_name=model_name, validate_flag=False)
            keyword_search = question.get_custom_context_keyword(k=10)
            question.fusion_context(semantic_search=semantic_search, keyword_search=keyword_search, k=5, semantic_weight=1.0, keyword_weight=1.5, model_name=model_name, validate_flag=False)
            
            preview["context"] = question.context
            
            normalized_schema, preview, llm_raw = analyze_xlsx(
                question=question,
                preview=preview,
                model_name=model_name,
            )
            
            end = time.time()
            print(f"[XLSX] Generation took {end-start:.2f} seconds")
            return normalized_schema, preview, llm_raw
        
        question = Query(query, [])

        query = question.question # Query 객체에서 원본 질문 문자열을 꺼내서 로컬 변수 query에 복사
        conciseprompt=f"""Rephrase the question to be clear and concise:
        
        {question.question}"""
        concisequery = submit_prompt_flex(conciseprompt, model=model_name).rstrip('"') # 질문을 더 간단하고 명확하게
        print(concisequery)
        question.query = concisequery # 약어, 통신 표준 용어가 붙음.

        question.def_TA_question() # 질문을 주제 분류용으로 정규화/보정 : 
        print()
        print('#'*50)
        print(query)
        print('#'*50)
        print()

        # question.get_3GPP_context(k=10, model_name=model_name, validate_flag=False) # 근거 문맥을 3GPP 문서에서 뽑아오는 단계
        semantic_search = question.get_custom_context(k=10, model_name=model_name, validate_flag=False)
        keyword_search= question.get_custom_context_keyword(k=10)
        question.fusion_context(semantic_search = semantic_search, keyword_search = keyword_search, model_name=model_name, validate_flag=False)
        
        '''loop = asyncio.get_event_loop()
        context_3gpp_future = loop.run_in_executor(None, question.get_custom_context, 10, model_name, False, False)
        online_info = await question.get_online_context(model_name=model_name, validator_flag=False)
        await context_3gpp_future
        for online_parag in online_info:
            question.context.append(online_parag)'''
        
        if answer is not None:
            response, response_raw, context , _ = check_question(question, answer, options, model_name=model_name) # 컨텍스트와 옵션을 포함한 프롬프트를 만들어 LLM에 답을 생성
            print(response_raw)
            end=time.time()
            print(f'Generation of this response took {end-start} seconds')
            return response, question.context, question.context_score
        elif options is not None:
            response, context , _ = check_question(question, answer, options, model_name=model_name) # 컨텍스트와 옵션을 포함한 프롬프트를 만들어 LLM에 답을 생성
            print(response)
            end=time.time()
            print(f'Generation of this response took {end-start} seconds')
            return response, question.context, question.context_score
        else:
            response, context, _ = generate(question, model_name)
            end=time.time()
            print(f'Generation of this response took {end-start} seconds')
            return response, question.context, question.context_score
   
    except Exception as e:
        print(f"An error occurred: {e}")
        print(traceback.format_exc())

"""
if __name__ == "__main__":
    question =  {
        "question": "In supporting an MA PDU Session, what does Rel-17 enable in terms of 3GPP access over EPC? [3GPP Release 17]",
        "options" : { 
        "option 1": "Direct connection of 3GPP access to 5GC",
        "option 2": "Establishment of user-plane resources over EPC",
        "option 3": "Use of NG-RAN access for all user-plane traffic",
        "option 4": "Exclusive use of a non-3GPP access for user-plane traffic"
        },
        "answer": "option 2: Establishment of user-plane resources over EPC",
        "explanation": "Rel-17 enables the establishment of user-plane resources over EPC for 3GPP access in supporting an MA PDU Session, allowing for simultaneous traffic over EPC and non-3GPP access.",
        "category": "Standards overview"
    }
    # Example using an MCQ
    response, context = TelcoRAG(question['question'], question['answer'], question['options'], model_name='Qwen/Qwen3-32B' )
    print(f"[Response]: {response}", '\n')
    # Example using an open-end question           
    # response, context = TelcoRAG(question['question'], model_name='/NAS/inno_aidev/local_models/Qwen2.5-Coder-7B-Instruct/' )
    # print(response, '\n')
"""
    
#"""

if __name__ == "__main__":
    import argparse

    base_dir = Path(__file__).resolve().parent

    ap = argparse.ArgumentParser()
    ap.add_argument("--query", type=str, default=None)
    ap.add_argument("--answer", type=str, default=None)
    ap.add_argument("--options", type=str, default=None, help="JSON string for options dict")
    ap.add_argument("--model-name", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--xlsx-input-file", "-xif", type=str, default=None, help="Single .xlsx file path")
    ap.add_argument("--xlsx-input-dir", "-xid", type=str, default=None, help="Directory containing .xlsx files")
    ap.add_argument("--xlsx-input-glob", type=str, default="*.xlsx", help="Glob pattern for xlsx files")
    ap.add_argument("--xlsx-output-dir", "-xod", type=str, default=str(base_dir / "outputs"), help="Output directory for xlsx mode")
    ap.add_argument("--max-sample-rows", type=int, default=2000, help="Max non-empty samples per column for LLM prompt")
    ap.add_argument("--max-scan-rows", type=int, default=2000, help="Max rows scanned from xlsx")
    ap.add_argument("--without-RAG", action="store_true", help="LLM only mode")
    args = ap.parse_args()

    def parse_options(options_str):
        if options_str is None:
            return None
        options_str = options_str.strip()
        if not options_str:
            return None
        try:
            parsed = jsonlib.loads(options_str)
        except Exception:
            parsed = None

        if isinstance(parsed, dict):
            items = []
            for k, v in parsed.items():
                if v is None or v == "":
                    items.append(str(k))
                else:
                    items.append(f"{k}: {v}")
            return items
        if isinstance(parsed, (list, tuple, set)):
            return list(parsed)
        if isinstance(parsed, str):
            return [parsed]

        # Fallback for non-JSON inputs like:
        # {"option 1: ...", "option 2: ..."}
        import re
        extracted = re.findall(r'"([^"]+)"', options_str)
        if extracted:
            return extracted
        return [options_str]

    def _to_jsonable(obj):
        if isinstance(obj, set):
            return sorted(obj)
        if isinstance(obj, dict):
            return {k: _to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_jsonable(v) for v in obj]
        return obj

    def run_once(query, answer, options):
        if not query:
            return

        response, context, context_score = asyncio.run(TelcoRAG(
            query=query,
            answer=answer,
            options=options,
            model_name=args.model_name
        ))
        
        output_dir = os.path.join(str(base_dir / "outputs"), "pipeline_online")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"response_context_{int(time.time() * 1000)}.json")
        output = {
            "query": query,
            "answer": answer,
            "options": _to_jsonable(options),
            "response": _to_jsonable(response),
            "context": _to_jsonable(context),
            "context_score" : _to_jsonable(context_score),
        }
        with open(output_path, "w") as f:
            jsonlib.dump(output, f, indent=4)
        
        print("-" * 50)
        print(f"[Response]:\n{response}")
        print("-" * 50 + "\n")

    def resolve_xlsx_inputs() -> list[str]:
        if args.xlsx_input_file:
            input_path = Path(args.xlsx_input_file)
            if input_path.suffix.lower() != ".xlsx":
                raise ValueError(f"Only .xlsx is supported: {input_path}")
            if not input_path.exists():
                raise FileNotFoundError(f"Input file not found: {input_path}")
            return [str(input_path)]

        if args.xlsx_input_dir:
            input_dir = Path(args.xlsx_input_dir)
        else:
            input_dir = base_dir / "inputs"
            if not input_dir.exists():
                return []

        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        matched = sorted(input_dir.glob(args.xlsx_input_glob))
        return [str(path) for path in matched if path.is_file() and path.suffix.lower() == ".xlsx"]

    def run_xlsx_once(xlsx_path: str):
        start = time.time()
        response, preview, llm_raw = asyncio.run(TelcoRAG(
            query="",
            xlsx_file=xlsx_path,
            model_name=args.model_name,
            max_sample_rows=args.max_sample_rows,
            max_scan_rows=args.max_scan_rows,
        ))

        output_dir = Path(args.xlsx_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{Path(xlsx_path).stem}.json"
        with output_path.open("w", encoding="utf-8") as f:
            jsonlib.dump(_to_jsonable(response), f, indent=2)

        end = time.time()
        print(
            f"[XLSX] {xlsx_path} -> {output_path} "
            f"(columns={preview['column_count']}, scanned_rows={preview['rows_scanned']}, {end - start:.2f}s)"
        )

    print("=== START ===")
    
    xlsx_mode_requested = bool(args.xlsx_input_file or args.xlsx_input_dir)
    if xlsx_mode_requested:
        xlsx_targets = resolve_xlsx_inputs()
        if not xlsx_targets:
            print("[XLSX] No .xlsx files found.")
        for xlsx_path in xlsx_targets:
            try:
                run_xlsx_once(xlsx_path)
            except Exception as e:
                print(f"[XLSX][ERROR] {xlsx_path}: {e}")
                traceback.print_exc()
                
    elif args.query is not None:
        opts = parse_options(args.options)
        run_once(args.query, args.answer, opts)
        
    else:
        while True:
            try:
                user_query = input("User Query: ").strip()
                run_once(user_query, None, None)
            except KeyboardInterrupt:
                print("\n\ndetect interrupt. program exitted.")
                break
            except Exception as e:
                print(f"\nERROR: {e}")
                traceback.print_exc()
#"""
