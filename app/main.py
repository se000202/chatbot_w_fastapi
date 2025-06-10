# ✅ FastAPI 최종본 — /chat + /chat_stream 분리

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

def process_line_with_latex_safe(content):
    # LaTeX 수식 패턴: $$ ... $$
    pattern = re.compile(r'(\$\$.*?\$\$)', re.DOTALL)

    parts = pattern.split(content)

    result_parts = []
    for part in parts:
        if part.startswith('$$') and part.endswith('$$'):
            # LaTeX 수식 부분 → 그대로 둠
            result_parts.append(part)
        else:
            # 일반 텍스트 → \n → \n\n 변환
            part = part.replace('\n', '\n\n')
            # 중복 줄바꿈은 최소화
            part = re.sub(r'\n\n\n+', '\n\n', part)
            result_parts.append(part)

    return ''.join(result_parts)



# GPT Streaming generator
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
            yield process_line_with_latex_safe(content)



# Safe eval 계산
forbidden_keywords = ["import", "def", "exec", "eval", "os.", "__"]

def compute_expression(expr: str) -> str:
    try:
        if any(keyword in expr for keyword in forbidden_keywords):
            return f"계산 중 오류 발생: 금지된 표현식이 감지되었습니다."

        safe_globals = {
            "__builtins__": {},
            "sum": sum,
            "range": range,
            "prod": prod,
            "round": round,
            "reduce": reduce,
            "all": all,
            "int": int,
            "float": float,
            "abs": abs,
            "pow": pow,
            "math": math,
            "sqrt": math.sqrt,
            "log": math.log,
            "log10": math.log10,
            "exp": math.exp
        }
        result = eval(expr, safe_globals)
        return f"계산 결과: {result}"
    except Exception as e:
        return f"계산 중 오류 발생: {e}"

# Improved auto_wrap_latex
def auto_wrap_latex(response: str) -> str:
    if "$$" in response:
        return response

    environments = [
        "align\\*", "align", "equation", "gather", "multline"
    ]

    response = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', response, flags=re.DOTALL)

    for env in environments:
        pattern = rf'\\begin{{{env}}}(.*?)\\end{{{env}}}'
        response = re.sub(pattern, r'$$\1$$', response, flags=re.DOTALL)

    if len(response.strip()) > 300:
        return response

    formula_keywords = ["=", "+", "-", "*", "/", "^", "\\approx","\\sqrt", "\\frac", "\\sum", "\\int", "\\log", "\\sin", "\\cos", "\\tan", "\\text", "\\displaystyle"]

    if any(keyword in response for keyword in formula_keywords):
        if "\n" not in response and all(c.isalnum() or c in " +-*/^=()[]{}\\._" for c in response.strip()):
            print("[DEBUG] Auto-wrapping response as LaTeX (fallback keyword match).")
            return f"$$ {response.strip()} $$"

    return response

# 추가: get_chatbot_response 함수 정의
def get_chatbot_response(messages):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    return response.choices[0].message.content.strip()

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    messages = req.messages

    user_msgs = [m["content"] for m in messages if m["role"] == "user"]
    last_msg = user_msgs[-1] if user_msgs else ""

    calc_keywords = ["합", "곱", "피보나치", "product of primes", "sum of primes", "fibonacci"]

    if any(keyword in last_msg for keyword in calc_keywords):
        system_prompt = [
            {"role": "system", "content": "You are an assistant that converts calculation requests into ONE-LINE Python expressions. "
                                          "You must NOT define functions. You must NOT use 'is_prime' or any undefined functions. "
                                          "You must NOT use import statements. You must NOT use eval or exec or os. "
                                          "You must use list comprehension with 'all(x % d != 0 ...)' inline to detect primes. "
                                          "If the user asks for sum of primes, output 'sum([...])'. "
                                          "If the user asks for product of primes, output 'prod([...])'. "
                                          "If the user asks for the nth Fibonacci number, you MUST use a one-line expression with 'reduce' only. "
                                          "Only output the expression and nothing else."},
            {"role": "user", "content": last_msg}
        ]
        expr = get_chatbot_response(system_prompt)
        result = compute_expression(expr)
        return {"response": result}

    system_prompt_default = [
        {"role": "system", "content": "You are a helpful assistant. "
                                      "If your output includes a mathematical formula or expression, always surround it with $$...$$."
                                      "Do NOT use \\( ... \\) or \\[ ... \\]. Only use $$...$$ to enclose math."
                                      "If your output is normal text, do not use $$."
                                      Always use double line breaks (\\n\\n) between paragraphs, lists, and after formulas for readability."},
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=system_prompt_default + messages
    )
    return {"response": response.choices[0].message.content.strip()}

@app.post("/chat_stream")
async def chat_stream_endpoint(req: ChatRequest):
    messages = req.messages

    user_msgs = [m["content"] for m in messages if m["role"] == "user"]
    last_msg = user_msgs[-1] if user_msgs else ""

    calc_keywords = ["합", "곱", "피보나치", "product of primes", "sum of primes", "fibonacci"]

    if any(keyword in last_msg for keyword in calc_keywords):
        system_prompt = [
            {"role": "system", "content": "You are an assistant that converts calculation requests into ONE-LINE Python expressions. "
                                          "You must NOT define functions. You must NOT use 'is_prime' or any undefined functions. "
                                          "You must NOT use import statements. You must NOT use eval or exec or os. "
                                          "You must use list comprehension with 'all(x % d != 0 ...)' inline to detect primes. "
                                          "If the user asks for sum of primes, output 'sum([...])'. "
                                          "If the user asks for product of primes, output 'prod([...])'. "
                                          "If the user asks for the nth Fibonacci number, you MUST use a one-line expression with 'reduce' only. "
                                          "Only output the expression and nothing else."},
            {"role": "user", "content": last_msg}
        ]
        expr = get_chatbot_response(system_prompt)
        result = compute_expression(expr)
        return {"response": result}

    system_prompt_default = [
        {"role": "system", "content": "You are a helpful assistant. "
                                      "If your output includes a mathematical formula or expression, always surround it with $$...$$."
                                      "Do NOT use \\( ... \\) or \\[ ... \\]. Only use $$...$$ to enclose math."
                                      "If your output is normal text, do not use $$."
                                      Always use double line breaks (\\n\\n) between paragraphs, lists, and after formulas for readability."},
    ]

    return StreamingResponse(
        gpt_stream(system_prompt_default + messages),
        media_type="text/plain"
    )
