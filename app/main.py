from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict
import os
from dotenv import load_dotenv
from openai import OpenAI
import math
import re
from functools import reduce

# Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]

# ---- LaTeX 후처리 ----
def auto_wrap_inline_latex(response: str) -> str:
    inline_latex_pattern = re.compile(r'(\\(?:frac|sqrt|sum|int|log|sin|cos|tan)[^$ \n]*)')

    def replacer(match):
        return f'$$ {match.group(1)} $$'

    response = inline_latex_pattern.sub(replacer, response)
    return response

def auto_wrap_list_latex(response: str) -> str:
    lines = response.split('\n')
    new_lines = []
    for line in lines:
        if re.match(r'^\s*[-*]\s', line) and re.search(r'\\(frac|sqrt|sum|int|log|sin|cos|tan)', line):
            content = line.strip()
            content_no_bullet = re.sub(r'^[-*]\s+', '', content)
            new_lines.append(f'- $$ {content_no_bullet} $$')
        else:
            new_lines.append(line)
    return '\n'.join(new_lines)

# ---- GPT 호출 ----
def get_chatbot_response(messages):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    return response.choices[0].message.content.strip()

# Streaming generator
def gpt_stream(messages):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        stream=True
    )
    for chunk in response:
        delta = chunk.choices[0].delta
        content = getattr(delta, "content", None)
        if content:
            content = auto_wrap_inline_latex(content)
            content = auto_wrap_list_latex(content)
            yield content

# ---- 안전한 Python 실행 ----
forbidden_keywords = ["import os", "import sys", "import subprocess", "import shutil", "import pathlib", "open(", "eval", "exec", "__", "os.", "sys.", "subprocess."]

def safe_exec_function(code: str) -> str:
    try:
        # 금지된 키워드 검사
        if any(keyword in code for keyword in forbidden_keywords):
            return f"🚫 위험한 코드 감지됨: 실행 차단됨."

        # 안전한 글로벌 변수
        safe_globals = {
            "__builtins__": {},
            "math": math,
            "sum": sum,
            "range": range,
            "prod": math.prod if hasattr(math, "prod") else None,
            "reduce": reduce,
            "all": all,
            "int": int,
            "float": float,
            "abs": abs,
            "pow": pow
        }
        safe_locals = {}

        exec(code, safe_globals, safe_locals)

        # compute() 함수가 정의되어 있으면 실행
        if "compute" in safe_locals and callable(safe_locals["compute"]):
            result = safe_locals["compute"]()
            return f"계산 결과: {result}"
        else:
            return "✅ 코드 실행 성공. compute() 함수가 정의되지 않았습니다."
    except Exception as e:
        return f"🚫 코드 실행 중 오류 발생: {e}"

# ---- /chat endpoint ----
@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    messages = req.messages
    user_msgs = [m["content"] for m in messages if m["role"] == "user"]
    last_msg = user_msgs[-1] if user_msgs else ""

    # 🔥 코드 생성 요청인지 확인
    code_keywords = ["파이썬 코드", "python 코드", "Python function", "def compute", "코드 작성", "코드로 해결"]

    if any(keyword in last_msg for keyword in code_keywords):
        system_prompt = [
            {"role": "system", "content": "You are a helpful assistant. "
                                          "You must generate Python code to solve the user's request. "
                                          "You must define a function called compute() with no arguments. "
                                          "You can use the math module, and you do not need to import math explicitly — it is already provided. "
                                          "You may use math.sqrt(), math.log(), math.factorial(), math.sin(), math.cos(), etc. "
                                          "NEVER use os, sys, subprocess, pathlib, shutil, open(), eval(), exec(), __anything__. "
                                          "Always produce code that is safe to execute. "
                                          "Do NOT print the result inside compute(); just return the result."
                                          "Output only the Python code block."},
            {"role": "user", "content": last_msg}
        ]
        code = get_chatbot_response(system_prompt)
        result = safe_exec_function(code)
        return {"response": result}

    # Default prompt
    system_prompt_default = [
        {"role": "system", "content": "You are a helpful assistant. "
                                      "If your output includes a mathematical formula or expression, always surround it with $$...$$. "
                                      "Do NOT use \\( ... \\) or \\[ ... \\]. Only use $$...$$ to enclose math. "
                                      "If your output includes inline LaTeX expressions (\\frac, \\sqrt, \\sum, etc.) in lists or bullet points, also enclose the entire list item with $$...$$. "
                                      "If your output is normal text, do not use $$. "
                                      "If your output includes multiple paragraphs or lists, always use double line breaks (\\n\\n) for line breaks."},
    ]

    answer = get_chatbot_response(system_prompt_default + messages)
    answer = auto_wrap_inline_latex(answer)
    answer = auto_wrap_list_latex(answer)

    return {"response": answer}

# ---- /chat_stream endpoint ----
@app.post("/chat_stream")
async def chat_stream_endpoint(req: ChatRequest):
    messages = req.messages
    user_msgs = [m["content"] for m in messages if m["role"] == "user"]
    last_msg = user_msgs[-1] if user_msgs else ""

    code_keywords = ["파이썬 코드", "python 코드", "Python function", "def compute", "코드 작성", "코드로 해결"]

    if any(keyword in last_msg for keyword in code_keywords):
        system_prompt = [
            {"role": "system", "content": "You are a helpful assistant. "
                                          "You must generate Python code to solve the user's request. "
                                          "You must define a function called compute() with no arguments. "
                                          "You can use the math module, and you do not need to import math explicitly — it is already provided. "
                                          "You may use math.sqrt(), math.log(), math.factorial(), math.sin(), math.cos(), etc. "
                                          "NEVER use os, sys, subprocess, pathlib, shutil, open(), eval(), exec(), __anything__. "
                                          "Always produce code that is safe to execute. "
                                          "Do NOT print the result inside compute(); just return the result."
                                          "Output only the Python code block."},
            {"role": "user", "content": last_msg}
        ]
        code = get_chatbot_response(system_prompt)
        result = safe_exec_function(code)
        return {"response": result}

    system_prompt_default = [
        {"role": "system", "content": "You are a helpful assistant. "
                                      "If your output includes a mathematical formula or expression, always surround it with $$...$$. "
                                      "Do NOT use \\( ... \\) or \\[ ... \\]. Only use $$...$$ to enclose math. "
                                      "If your output includes inline LaTeX expressions (\\frac, \\sqrt, \\sum, etc.) in lists or bullet points, also enclose the entire list item with $$...$$. "
                                      "If your output is normal text, do not use $$. "
                                      "If your output includes multiple paragraphs or lists, always use double line breaks (\\n\\n) for line breaks."},
    ]

    return StreamingResponse(
        gpt_stream(system_prompt_default + messages),
        media_type="text/plain"
    )
