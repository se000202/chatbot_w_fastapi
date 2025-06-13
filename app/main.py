# ✅ FastAPI 최종본 — /chat + /chat_stream 분리 + safe_exec_function 통합 + LaTeX 후처리 포함

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict
import os
from dotenv import load_dotenv
from openai import OpenAI
from math import prod
from functools import reduce
import math
import re

# Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]

# ---- LaTeX 후처리 함수들 ----

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

def merge_consecutive_latex_blocks(response: str) -> str:
    # $$ ... $$$$ ... $$ → $$ ... ... $$ 형태로 병합
    response = re.sub(r'\$\$\s*\$\$', '', response)
    return response

# ---- GPT 호출 관련 ----

def get_chatbot_response(messages):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    return response.choices[0].message.content.strip()

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
            content = merge_consecutive_latex_blocks(content)
            yield content

# ---- 안전한 Python 실행 ----

forbidden_keywords = ["import", "exec", "eval", "os.", "__"]

def safe_exec_function(code: str) -> str:
    try:
        if any(keyword in code for keyword in forbidden_keywords):
            return "⚠️ 오류: 금지된 표현식이 감지되었습니다."

        safe_globals = {
            "__builtins__": {},
            "math": math,
            "sum": sum,
            "range": range,
            "prod": prod,
            "reduce": reduce,
            "all": all,
            "int": int,
            "float": float,
            "abs": abs,
            "pow": pow,
            "sqrt": math.sqrt,
            "log": math.log,
            "log10": math.log10,
            "exp": math.exp,
        }
        safe_locals = {}

        # print() 결과 캡처용
        from io import StringIO
        import sys

        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()

        exec(code, safe_globals, safe_locals)

        sys.stdout = old_stdout
        output = mystdout.getvalue()

        return output if output.strip() else "✅ 코드 실행 완료 (출력 없음)"
    except Exception as e:
        return f"⚠️ 실행 중 오류 발생: {e}"

# ---- /chat endpoint ----

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    messages = req.messages
    user_msgs = [m["content"] for m in messages if m["role"] == "user"]
    last_msg = user_msgs[-1] if user_msgs else ""

    calc_keywords = ["합", "곱", "피보나치", "product of primes", "sum of primes", "fibonacci", "python 코드"]

    if any(keyword in last_msg for keyword in calc_keywords):
        system_prompt = [
            {"role": "system", "content": "You are an assistant that generates SAFE Python code for mathematical calculations. "
                                          "NEVER use import statements. NEVER use exec, eval, os, subprocess, __, or system calls. "
                                          "ALLOWED functions: math, sum, range, prod, reduce, all, int, float, abs, pow, sqrt, log, log10, exp. "
                                          "You should only output a clean Python function or script that can be executed with exec(). "
                                          "ALWAYS use print() to show the final result of the calculation to the user. "
                                          "Example: print('Result:', value). DO NOT use return statements. DO NOT add explanations or markdown."},
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
                                      "If your output includes multiple paragraphs or lists, always use double line breaks (\\n\\n) for line breaks. "
                                      "Do not output any HTML or Javascript."},
    ]

    answer = get_chatbot_response(system_prompt_default + messages)
    answer = auto_wrap_inline_latex(answer)
    answer = auto_wrap_list_latex(answer)
    answer = merge_consecutive_latex_blocks(answer)

    return {"response": answer}

# ---- /chat_stream endpoint ----

@app.post("/chat_stream")
async def chat_stream_endpoint(req: ChatRequest):
    messages = req.messages
    user_msgs = [m["content"] for m in messages if m["role"] == "user"]
    last_msg = user_msgs[-1] if user_msgs else ""

    calc_keywords = ["합", "곱", "피보나치", "product of primes", "sum of primes", "fibonacci", "python 코드"]

    if any(keyword in last_msg for keyword in calc_keywords):
        system_prompt = [
            {"role": "system", "content": "You are an assistant that generates SAFE Python code for mathematical calculations. "
                                          "NEVER use import statements. NEVER use exec, eval, os, subprocess, __, or system calls. "
                                          "ALLOWED functions: math, sum, range, prod, reduce, all, int, float, abs, pow, sqrt, log, log10, exp. "
                                          "You should only output a clean Python function or script that can be executed with exec(). "
                                          "ALWAYS use print() to show the final result of the calculation to the user. "
                                          "Example: print('Result:', value). DO NOT use return statements. DO NOT add explanations or markdown."},
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
                                      "If your output includes multiple paragraphs or lists, always use double line breaks (\\n\\n) for line breaks. "
                                      "Do not output any HTML or Javascript."},
    ]

    return StreamingResponse(
        gpt_stream(system_prompt_default + messages),
        media_type="text/plain"
    )
