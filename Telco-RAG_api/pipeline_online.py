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
        # question.get_custom_context(k=10, model_name=model_name, validate_flag=False) # 근거 문맥을 3GPP 문서에서 뽑아오는 단계
        loop = asyncio.get_event_loop()
        context_3gpp_future = loop.run_in_executor(None, question.get_custom_context, 10, model_name, False, False)
        online_info = await question.get_online_context(model_name=model_name, validator_flag=False)
        await context_3gpp_future

        for online_parag in online_info:
            question.context.append(online_parag)
        
        if answer is not None:
            response, context , _ = check_question(question, answer, options, model_name=model_name) # 컨텍스트와 옵션을 포함한 프롬프트를 만들어 LLM에 답을 생성
            print(context)
            end=time.time()
            print(f'Generation of this response took {end-start} seconds')
            return response, question.context
        elif options is not None:
            response, context , _ = check_question(question, answer, options, model_name=model_name) # 컨텍스트와 옵션을 포함한 프롬프트를 만들어 LLM에 답을 생성
            print(context)
            end=time.time()
            print(f'Generation of this response took {end-start} seconds')
            return response, context
        else:
            response, context, _ = generate(question, model_name)
            end=time.time()
            print(f'Generation of this response took {end-start} seconds')
            return response, context
    
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
    print("=== START ===")
    print("for stopping, enter 'exit' or 'quit'\n")

    while True:
        try:
            # 사용자 입력 받기
            user_query = input("User Query: ").strip()

            # 종료 조건 확인
            if user_query.lower() in ['exit', 'quit']:
                print("program exitted.")
                break
            
            if not user_query:
                continue
            
            # Open-ended
            response, context = asyncio.run(TelcoRAG(
                query=user_query, 
                answer=None, 
                options=None, 
                model_name='Qwen/Qwen3-30B-A3B-Instruct-2507'
            ))
            
            os.makedirs("outputs", exist_ok=True)
            output_path = os.path.join("outputs", f"response_context_{int(time.time() * 1000)}.json")
            output = {
                "query": user_query,
                "response": response, 
                "context": context
            }
            with open(output_path, "w") as f:
                ujson.dump(output, f, indent=4)
            
            print("-" * 50)
            print(f"[Response]:\n{response}")
            print("-" * 50 + "\n")

        except KeyboardInterrupt:
            print("\n\ndetect interrupt. program exitted.")
            break
        except Exception as e:
            print(f"\nERROR: {e}")
            traceback.print_exc()
#"""
