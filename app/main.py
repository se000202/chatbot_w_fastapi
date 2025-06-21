from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import os
from dotenv import load_dotenv
from openai import OpenAI
import ast
import sys
import re
import requests
from bs4 import BeautifulSoup
from contextlib import redirect_stdout
import io

# Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]

def get_chatbot_response(messages):
    cleaned = [m for m in messages if m.get("content") is not None]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=cleaned
    )
    content = response.choices[0].message.content.strip()
    return content if content else ""

def clean_code_block(text: str) -> str:
    code_blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()

    lines = text.splitlines()
    code_start = -1
    for i, line in enumerate(lines):
        if line.strip().startswith("def "):
            code_start = i
            break
    if code_start != -1:
        return "\n".join(lines[code_start:]).strip()

    return text.strip()

def extract_called_functions(code: str) -> List[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    called_funcs = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called_funcs.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called_funcs.append(node.func.attr)
    return called_funcs

def safe_exec_function_with_trace(code: str) -> str:
    sys.set_int_max_str_digits(100000)
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"❌ 코드 문법 오류: {str(e)}"

    dangerous_nodes = {
        ast.Call: ['eval', 'exec', 'open', 'system', 'popen', 'spawn'],
        ast.Import: ['os', 'sys', 'subprocess'],
        ast.ImportFrom: ['os', 'sys', 'subprocess'],
        ast.Attribute: ['__import__', 'system', 'popen', 'spawn']
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in dangerous_nodes[ast.Call]:
                raise ValueError("위험합니다.")
            if isinstance(node.func, ast.Attribute) and node.func.attr in dangerous_nodes[ast.Call]:
                raise ValueError("위험합니다.")
        if isinstance(node, ast.Import):
            for name in node.names:
                if name.name in dangerous_nodes[ast.Import]:
                    raise ValueError("위험합니다.")
        if isinstance(node, ast.ImportFrom):
            if node.module in dangerous_nodes[ast.ImportFrom]:
                raise ValueError("위험합니다.")
        if isinstance(node, ast.Attribute):
            if node.attr in dangerous_nodes[ast.Attribute]:
                raise ValueError("위험합니다.")

    local_vars = {}
    output = io.StringIO()
    with redirect_stdout(output):
        try:
            exec(code, local_vars)
            if "main" not in local_vars:
                return "❌ main 함수가 정의되어 있지 않습니다."
            local_vars["main"]()
        except Exception as e:
            return f"❌ 실행 중 오류 발생: {str(e)}"
    result = output.getvalue().strip()
    called_funcs = extract_called_functions(code)
    trace_info = f"🧠 실행된 함수: {', '.join(set(called_funcs)) or '없음'}\n\n🖨️ 출력 결과: {result}"
    return trace_info

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    messages = req.messages
    user_msgs = [m["content"] for m in messages if m["role"] == "user"]
    last_msg = user_msgs[-1] if user_msgs else ""

    judge_prompt = [
        {"role": "system", "content": "You classify if the user's message can be solved using Python code. Respond with 'YES, ...' or 'NO!,'."},
        {"role": "user", "content": last_msg}
    ]
    judge_response = get_chatbot_response(judge_prompt)

    if judge_response and judge_response.strip().startswith("YES,"):
        task_description = judge_response.strip()[4:].strip()
        code_prompt = [
            {"role": "system", "content": """
You are a Python code generator. Generate a program that solves the given task.

Rules:
- Define a function named main()
- Call main() at the end
- Use print() inside main() for output
- Do not use eval, exec, os, __import__, or unsafe functions
"""},
            {"role": "user", "content": task_description}
        ]
        code_response = get_chatbot_response(code_prompt)
        cleaned_code = clean_code_block(code_response)

        if cleaned_code and "def main" in cleaned_code and "main(" in cleaned_code:
            try:
                result = safe_exec_function_with_trace(cleaned_code)
                return {"response": f"```python\n{cleaned_code}\n```\n\n{result}"}
            except Exception as e:
                return {"response": f"❌ 코드 실행 중 오류 발생: {str(e)}"}
        else:
            return {"response": "❌ GPT가 실행 가능한 코드를 생성하지 못했습니다."}

    else:
        general_prompt = [
            {"role": "system", "content": """
You are a helpful assistant.
If your output includes mathematical expressions, wrap them with $$...$$.
Use plain language and structured lists if needed.
"""},
            {"role": "user", "content": last_msg}
        ]
        general_response = get_chatbot_response(general_prompt)
        return {"response": general_response or "❌ GPT 응답이 null입니다. 다시 시도해주세요."}


# @app.post("/chat")
# async def chat_endpoint(req: ChatRequest):
#     messages = req.messages
#     user_msgs = [m["content"] for m in messages if m["role"] == "user"]
#     last_msg = user_msgs[-1] if user_msgs else ""
#     # Case 1: 문제의 정의 요청
#     system_prompt_task = [
#         {"role": "system", "content": """
#             You are professional python programmer which does the Following task.
#             Task 1 : When the chat is given from the user, you should determine whether given chat can be converted to the task,
#             which can be solved by using the python program.
#             If the given chat can be converted into task, return the string which initial three character is "YES," and continued string is 
#             the Task which the chat is converted.
#             Otherwise, return "NO!,"
#             """}
#         ]
#     code = get_chatbot_response(system_prompt_task + messages)
#     if code != "NO!":
#         system_prompt_python = [{"role": "system","contnet": """
#             You had recevied the string from User.
#             write the python program which solves the Task of the given string.
#             The program must include the function named "main" and the must call the function "main".
#             the main function must print the result of output which the task requires.
#             Do NOT use 'eval', 'exec', 'os', '__', or any unsafe functions.
#             You MUST NOT assign the result to a variable inside the function.
            
#             Example:
#             - Task: sum of integers between 1 and 10
#             - Example Respone:
#             def main(S,E):
#                 s = 0
#                 for i in range(S,E+1):
#                     s += i
#                 print(s)
#             main(1,10)
#             Note : 0,1 and Negative number is not a Prime.
#         """}                
#         ]
#         code = get_chatbot_response(system_prompt_python + messages)
#         code = clean_code_block(code)
#         try:
#             result = safe_exec_function(code)
#             return {"response": result}
#         except Exception as e:
#             return {"response": f"❌ 코드 실행 중 오류 발생: {str(e)}"}
#     else:
#         system_prompt_general = [
#                 {"role": "system", "content": """
#                 You are a helpful assistant. 
#                 If your output includes a mathematical formula or expression, always surround it with $$...$$.
#                 Do NOT use \\( ... \\) or \\[ ... \\]. Only use $$...$$ to enclose math.
#                 If your output is normal text, do not use $$.
#                 If your output includes multiple paragraphs or lists, always use double line breaks (\\n\\n) for line breaks.
#                 You are not a Python code generator unless specifically asked to write Python code.
#                 """}
#             ]
#         answer = get_chatbot_response(system_prompt_general + messages)
#         return {"response": answer}
