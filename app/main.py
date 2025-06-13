# ✅ FastAPI 최종본 — /chat 단일 endpoint + 수학문제 → Python 코드 → safe_exec + 일반 챗봇 처리

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict
import os
from dotenv import load_dotenv
from openai import OpenAI
import ast
import math
from math import prod
from functools import reduce

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

# 안전한 exec 처리
def safe_exec_function(code: str) -> str:
    try:
        # AST 검사
        tree = ast.parse(code)
        dangerous_nodes = {
            ast.Call: ['eval', 'exec', 'open', 'system', 'popen', 'spawn'],
            ast.Import: ['os', 'sys', 'subprocess'],
            ast.ImportFrom: ['os', 'sys', 'subprocess'],
            ast.Attribute: ['__import__', 'system', 'popen', 'spawn']
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name != "math":
                        raise ValueError(f"금지된 import 발견: {alias.name}")
            if isinstance(node, ast.ImportFrom):
                raise ValueError(f"금지된 ImportFrom 사용 발견.")
            if isinstance(node, ast.Attribute):
                if node.attr in dangerous_nodes[ast.Attribute]:
                    return False
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

        # 코드 실행
        exec(code, safe_globals, local_vars)

        # 결과 추출
        if "result" in local_vars:
            return f"계산 결과: {local_vars['result']}"
        else:
            return "코드 실행 완료 (결과 변수 'result' 없음)"
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
        # Case 1: Python 코드 생성
        system_prompt_math = [
            {"role": "system", "content": """
            You are a math assistant who writes correct Python code to solve the given math problem.
            Your goal is to output only the code (no explanations, no markdown).
            The code should be compatible with exec() and simple.
            Allowed imports: import math only.
            Do NOT use 'eval', 'exec', 'os', '__', or any unsafe functions.
            The output MUST be a valid Python code.
            You MUST assign the final result to a variable named 'result'.
            Example:
            result = sum([x for x in range(2, 100) if all(x % d != 0 for d in range(2, int(x**0.5)+1))])
            """}
        ]
        code = get_chatbot_response(system_prompt_math + messages)
        print(f"[DEBUG] Generated code: {repr(code)}")

        result = safe_exec_function(code)
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
