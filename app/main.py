# ✅ FastAPI 최종본 — /chat + /chat_stream 분리 + 안전한 LaTeX 후처리 추가

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

# 1️⃣ inline LaTeX 감싸기 (naked로 나오는 경우)
def auto_wrap_inline_latex(response: str) -> str:
    inline_latex_pattern = re.compile(r'(\\(?:frac|sqrt|sum|int|log|sin|cos|tan)[^$ \n]*)')

    def replacer(match):
        return f'$$ {match.group(1)} $$'

    response = inline_latex_pattern.sub(replacer, response)
    return response

# 2️⃣ list item 내에 LaTeX 있으면 전체 $$...$$ 감싸기
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

# ---- GPT 호출 관련 ----

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
            yield content

# ---- 계산 관련 ----

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

# ---- /chat endpoint ----

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

    # Default prompt (강화됨)
    system_prompt_default = [
        {"role": "system", "content": "You are a helpful assistant. "
                                      "If your output includes a mathematical formula or expression, always surround it with $$...$$."
                                      "Do NOT use \\( ... \\) or \\[ ... \\]. Only use $$...$$ to enclose math."
                                      "If your output includes inline LaTeX expressions (\\frac, \\sqrt, \\sum, etc.) in lists or bullet points, also enclose the entire list item with $$...$$."
                                      "If your output is normal text, do not use $$."
                                      "If your output includes multiple paragraphs or lists, always use double line breaks (\\n\\n) for line breaks."},
    ]

    answer = get_chatbot_response(system_prompt_default + messages)

    # 후처리 적용
    answer = auto_wrap_inline_latex(answer)
    answer = auto_wrap_list_latex(answer)

    return {"response": answer}

# ---- /chat_stream endpoint ----

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

    # Default prompt (강화됨)
    system_prompt_default = [
        {"role": "system", "content": "You are a helpful assistant. "
                                      "If your output includes a mathematical formula or expression, always surround it with $$...$$."
                                      "Do NOT use \\( ... \\) or \\[ ... \\]. Only use $$...$$ to enclose math."
                                      "If your output includes inline LaTeX expressions (\\frac, \\sqrt, \\sum, etc.) in lists or bullet points, also enclose the entire list item with $$...$$."
                                      "If your output is normal text, do not use $$."
                                      "If your output includes multiple paragraphs or lists, always use double line breaks (\\n\\n) for line breaks."},
    ]

    return StreamingResponse(
        gpt_stream(system_prompt_default + messages),
        media_type="text/plain"
    )
