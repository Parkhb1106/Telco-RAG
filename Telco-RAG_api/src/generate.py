import re
import traceback
from src.LLMs.LLM import submit_prompt_flex
from src.xlsx_schema import (
    _build_column_schema_prompt,
    _build_schema_prompt,
    _build_summary_prompt,
    _load_json_object,
    _normalize_schema,
)
import logging

def generate(question, model_name, yes_rag=True):
    """Generate a response using the GPT model given a structured question object."""
    try:
        # Constructing the content context from the question object
        context_section = ""
        if yes_rag:
            content = "\n".join(question.context)
            context_section = (
                "Considering the following context:\n"
                f"{content}\n\n"
                "Please answer the following question, add between paranthesis the retrieval(e.g. Retrieval 3) that you used for each eleement of your reasoning:\n"
                f"{question.question}"
            )
        
        prompt = f"""
        Please answer the following question:
        {question.query}

        {context_section}
        """
        print(prompt)
        logging.info("Generated system prompt for OpenAI completion.")
        
        predicted_answers_str = submit_prompt_flex(prompt, model=model_name)
        logging.info("Model response generated successfully.")

        if yes_rag:
            context = f"The retrieved context provided to the LLM is:\n{content}"
        else:
            context = ""
        return predicted_answers_str, context, question.question
        

    except Exception as e:
        # Logging the error and returning a failure indicator
        logging.error(f"An error occurred: {e}")
        return None, None, None
    
def find_option_number(text):
    """
    Finds all the occurrences of numbers preceded by the word 'option' in a given text.

    Parameters:
    - text: The text to search for 'option' followed by numbers.

    Returns:
    - A list of strings, each representing a number found after 'option'. The numbers are returned as strings.
    - If no matches are found, an empty list is returned.
    """
    try:
        text =  text.lower()
        # Define a regular expression pattern to find 'option' followed by non-digit characters (\D*), 
        # and then one or more digits (\d+)
        pattern = r'option\D*(\d+)'
        # Find all matches of the pattern in the text
        matches = re.findall(pattern, text)
        return matches  # Return the list of found numbers as strings
    except Exception as e:
        print(f"An error occurred while trying to find option numbers in the text: {e}")
        return []


def check_question(question, answer, options, model_name='gpt-4o-mini', yes_rag=True):
    """
    This function checks if the answer provided for a non-JSON formatted question is correct. 
    It dynamically selects the model based on the model_name provided and constructs a prompt 
    for the AI model to generate an answer. It then compares the generated answer with the provided 
    answer to determine correctness.

    Parameters:
    - question: A dictionary containing the question, options, and context.
    - model_name: Optional; specifies the model to use. Defaults to 'mistralai/Mixtral-8x7B-Instruct-v0.1' 
    if not provided or if the default model is indicated.

    Returns:
    - A tuple containing the updated question dictionary and a boolean indicating correctness.
    """
    try:
        # Normalize options to a list of display strings.
        if isinstance(options, dict):
            options_list = []
            for k, v in options.items():
                if v is None or v == "":
                    options_list.append(str(k))
                else:
                    options_list.append(f"{k}: {v}")
        elif isinstance(options, (list, tuple, set)):
            options_list = list(options)
        elif isinstance(options, str):
            options_list = [options]
        else:
            options_list = []

        options_text = '\n'.join(options_list)

        context_section = ""
        if yes_rag:
            content = "\n".join(question.context)
            context_section = (
                "Considering the following context:\n"
                f"{content}\n\n"
                "Please provide the answers to the following multiple choice question.\n"
                f"{question.question}\n\n"
            )
    
        syst_prompt = f"""
        Please provide the answers to the following multiple choice question.
        {question.query}
        
        {context_section}Options:
        Write only the option number corresponding to the correct answer:\n{options_text}
        
        Answer format should be: Answer option <option_id>
        """
        print(syst_prompt)
        # Generating the model's response based on the constructed prompt.
        predicted_answers_str = submit_prompt_flex(syst_prompt, model=model_name)
        predicted_answers_str = predicted_answers_str.replace('"\n', '",\n')
        print(predicted_answers_str)
        
        if yes_rag:
            context = f"The retrieved context provided to the LLM is:\n{content}"
        else:
            context = ""
        
        if answer is None:
            return predicted_answers_str, context, question.question
        
        print(answer)

        # Finding and comparing the predicted answer to the actual answer.
        answer_id = find_option_number(predicted_answers_str)
        real_answer_id = find_option_number(answer)

        if real_answer_id == answer_id:
            print("Correct\n")
            return  True, f"Option {answer_id}", context, syst_prompt
        else:
            print("Wrong\n")
            print(f"The chosen one is {answer_id}, but the answer is {real_answer_id}")
            return  False, f"Option {answer_id}", context, syst_prompt
    except Exception as e:
        # Error handling to catch and report errors more effectively.
        print(f"An error occurred: {e}")
        print(traceback.format_exc())
        return None, False
    
def analyze_xlsx(question, preview, model_name='gpt-4o-mini'):
    try:
        prompt = _build_schema_prompt(question, preview)
        llm_raw = submit_prompt_flex(prompt, model=model_name, output_json=True)
        parsed = _load_json_object(llm_raw)

        fallback_summary = (
            f'{preview["file_name"]} contains {preview["column_count"]} columns. '
            "This summary was auto-generated because model output did not include a summary."
        )
        normalized = _normalize_schema(parsed, preview["column_names"], fallback_summary)
        return normalized, preview, llm_raw
    except Exception as e:
        # Error handling to catch and report errors more effectively.
        print(f"An error occurred: {e}")
        print(traceback.format_exc())
        return None, False


def analyze_xlsx_column(question, preview, column_name, model_name='gpt-4o-mini', yes_rag=True):
    try:
        column = next((c for c in preview.get("columns", []) if c.get("name") == column_name), None)
        if column is None:
            column = {"name": column_name, "samples": []}

        prompt = _build_column_schema_prompt(question, preview, column, yes_rag)
        llm_raw = submit_prompt_flex(prompt, model=model_name, output_json=True)
        parsed = _load_json_object(llm_raw)

        if column_name in parsed and isinstance(parsed[column_name], dict):
            parsed = parsed[column_name]

        normalized = _normalize_schema(
            raw_schema={column_name: parsed if isinstance(parsed, dict) else {}},
            column_names=[column_name],
            fallback_summary="N/A",
        )
        return normalized[column_name], llm_raw
    except Exception as e:
        print(f"An error occurred: {e}")
        print(traceback.format_exc())
        normalized = _normalize_schema(
            raw_schema={},
            column_names=[column_name],
            fallback_summary="N/A",
        )
        return normalized[column_name], ""


def summarize_xlsx(question, preview, column_schema, model_name='gpt-4o-mini', yes_rag=True):
    try:
        prompt = _build_summary_prompt(question, preview, column_schema, yes_rag)
        llm_raw = submit_prompt_flex(prompt, model=model_name, output_json=True)
        parsed = _load_json_object(llm_raw)
        summary = str(parsed.get("summary", "")).strip()
        if not summary:
            summary = (
                f'{preview["file_name"]} contains {preview["column_count"]} columns. '
                "This summary was auto-generated because model output did not include a summary."
            )
        return summary, llm_raw
    except Exception as e:
        print(f"An error occurred: {e}")
        print(traceback.format_exc())
        fallback = (
            f'{preview["file_name"]} contains {preview["column_count"]} columns. '
            "This summary was auto-generated because summary generation failed."
        )
        return fallback, ""
