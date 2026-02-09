import os
import traceback
from src.query import Query
from src.generate import generate, check_question
from src.LLMs.LLM import submit_prompt_flex
import traceback
import git
import asyncio
import time
import ujson

folder_url = "https://huggingface.co/datasets/netop/Embeddings3GPP-R18"
clone_directory = "./3GPP-Release18"

if not os.path.exists(clone_directory):
    git.Repo.clone_from(folder_url, clone_directory)
    print("Folder cloned successfully!")
else:
    print("Folder already exists. Skipping cloning.")

async def TelcoRAG(query, answer= None, options= None, model_name='gpt-4o-mini'):
    try:
        start =  time.time()
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

    ap = argparse.ArgumentParser()
    ap.add_argument("--query", type=str, default=None)
    ap.add_argument("--answer", type=str, default=None)
    ap.add_argument("--options", type=str, default=None, help="JSON string for options dict")
    ap.add_argument("--model-name", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    args = ap.parse_args()

    def parse_options(options_str):
        if options_str is None:
            return None
        options_str = options_str.strip()
        if not options_str:
            return None
        try:
            parsed = ujson.loads(options_str)
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

    def run_once(query, answer, options):
        if not query:
            return

        def _to_jsonable(obj):
            if isinstance(obj, set):
                return sorted(obj)
            if isinstance(obj, dict):
                return {k: _to_jsonable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_to_jsonable(v) for v in obj]
            return obj
        
        response, context, context_score = asyncio.run(TelcoRAG(
            query=query,
            answer=answer,
            options=options,
            model_name=args.model_name
        ))
        
        output_dir = os.path.join("outputs", "pipeline_online")
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
            ujson.dump(output, f, indent=4)
        
        print("-" * 50)
        print(f"[Response]:\n{response}")
        print("-" * 50 + "\n")

    print("=== START ===")
    if args.query is not None:
        opts = parse_options(args.options)
        run_once(args.query, args.answer, opts)
    else:
        while True:
            try:
                user_query = input("User Query: ").strip()
                '''opts = [
                    "option 1: Direct connection of 3GPP access to 5GC",
                    "option 2: Establishment of user-plane resources over EPC",
                    "option 3: Use of NG-RAN access for all user-plane traffic",
                    "option 4: Exclusive use of a non-3GPP access for user-plane traffic"
                    ]
                answer = "option 2: Establishment of user-plane resources over EPC"'''
                run_once(user_query, None, None)
            except KeyboardInterrupt:
                print("\n\ndetect interrupt. program exitted.")
                break
            except Exception as e:
                print(f"\nERROR: {e}")
                traceback.print_exc()
#"""
