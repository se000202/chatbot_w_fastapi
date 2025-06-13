# ✅ FastAPI 최종본 — /chat 단일 endpoint + 함수 기반 safe_exec + args 자동 파싱 + 일반 챗봇 처리

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import os
from dotenv import load_dotenv
from openai import OpenAI
import ast
import math
from math import prod
from functools import reduce
import re  # 추가

# Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]

# GPT 호출
def get_chatbot_response(messages):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    return response.choices[0].message.content.strip()

# 숫자 자동 파싱 함수
def extract_numbers(text: str) -> List[float]:
    # 정수 또는 소수 모두 추출
    matches = re.findall(r'-?\d+\.?\d*', text)
    numbers = [float(m) if '.' in m else int(m) for m in matches]
    return numbers

# 안전한 exec 처리 (함수 정의 후 별도 호출)
def safe_exec_function(code: str, args: List) -> str:
    try:
        # AST 검사
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name != "math":
                        raise ValueError(f"금지된 import 발견: {alias.name}")
            if isinstance(node, ast.ImportFrom):
                raise ValueError(f"금지된 ImportFrom 사용 발견.")
            if isinstance(node, ast.Attribute):
                if node.attr in ['__import__', 'system', 'popen', 'spawn']:
                    raise ValueError(f"금지된 속성 사용 발견: {node.attr}")

        # 안전한 환경 구성
        safe_globals = {
            "__builtins__": {},
            "math": math,
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
        }

        local_vars = {}

        # 함수 정의 실행
        exec(code, safe_globals, local_vars)

        # 함수 호출
        if "f" not in local_vars:
            raise ValueError("함수 f 가 정의되어 있지 않습니다!")

        func = local_vars["f"]
        result = func(*args)

        return f"계산 결과: {result}"

    except Exception as e:
        return f"코드 실행 중 오류 발생: {e}"

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    messages = req.messages
    user_msgs = [m["content"] for m in messages if m["role"] == "user"]
    last_msg = user_msgs[-1] if user_msgs else ""

    # Case 1 → 수학/코드 관련 keywords
    calc_keywords = [
        "합", "곱", "피보나치", "피보나치 수", "product of primes", "sum of primes",
        "fibonacci", "소수", "소수의 합", "소수의 곱", "prime",
        "표준편차", "분산", "평균", "median", "variance", "standard deviation"
    ]

    if any(keyword in last_msg for keyword in calc_keywords):
        # Case 1: Python 함수 정의 요청
        system_prompt_math = [
            {"role": "system", "content": """
            You are a math assistant who writes correct Python code to solve the given math problem.

            Your goal is to output only the function definition (no explanations, no markdown, no variable assignment).
            The function name MUST be 'f'.
            The function must take one or more arguments, depending on the problem.
            Do NOT call the function.
            Do NOT assign the result to a variable.
            Allowed imports: import math only.
            Do NOT use 'eval', 'exec', 'os', '__', or any unsafe functions.

            You MUST NOT assign the result to a variable inside the function.

            Example:
            def f(a, b):
                return a + b
            """}
        ]
        code = get_chatbot_response(system_prompt_math + messages)
        print(f"[DEBUG] Generated code: {repr(code)}")

        # 🟡 자동으로 유저 입력에서 숫자 파싱
        args = extract_numbers(last_msg)
        print(f"[DEBUG] Extracted args: {args}")

        result = safe_exec_function(code, args)
        return {"response": result}

    else:
        # Case 2: 일반 챗봇
        system_prompt_general = [
            {"role": "system", "content": """
            You are a helpful assistant. 
            If your output includes a mathematical formula or expression, always surround it with $$...$$.
            Do NOT use \\( ... \\) or \\[ ... \\]. Only use $$...$$ to enclose math.
            If your output is normal text, do not use $$.
            If your output includes multiple paragraphs or lists, always use double line breaks (\\n\\n) for line breaks.
            You are not a Python code generator unless specifically asked to write Python code.
            """}
        ]
        answer = get_chatbot_response(system_prompt_general + messages)
        return {"response": answer}
