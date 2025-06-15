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
import sys

# Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]

# GPT 호출
def get_chatbot_response(messages):
    response = client.chat.completions.create(
        model="gpt-4o",
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
def safe_exec_function(code: str) -> str:
        sys.set_int_max_str_digits(10000)
        tree = ast.parse(code)
        
        # Define dangerous nodes to check for
        dangerous_nodes = {
            ast.Call: ['eval', 'exec', 'open', 'system', 'popen', 'spawn'],
            ast.Import: ['os', 'sys', 'subprocess'],
            ast.ImportFrom: ['os', 'sys', 'subprocess'],
            ast.Attribute: ['__import__', 'system', 'popen', 'spawn']
        }
        
        for node in ast.walk(tree):
            # Check for dangerous function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in dangerous_nodes[ast.Call]:
                       raise ValueError("위험합니다.")
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr in dangerous_nodes[ast.Call]:
                        raise ValueError("위험합니다.")
            
            # Check for dangerous imports
            if isinstance(node, ast.Import):
                for name in node.names:
                    if name.name in dangerous_nodes[ast.Import]:
                        raise ValueError("위험합니다.")
            
            # Check for dangerous from imports
            if isinstance(node, ast.ImportFrom):
                if node.module in dangerous_nodes[ast.ImportFrom]:
                    raise ValueError("위험합니다.")
            
            # Check for dangerous attributes
            if isinstance(node, ast.Attribute):
                if node.attr in dangerous_nodes[ast.Attribute]:
                    raise ValueError("위험합니다.")

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
            "len": len
        }

        local_vars = {}

        # 함수 정의 실행
        exec(code, safe_globals, local_vars)

        # 함수 호출
        if "f" not in local_vars:
            raise ValueError("함수 f 가 정의되어 있지 않습니다!")

        func = local_vars["f"]
        result = func()

        return f"계산 결과: {result}"

def clean_code_block(code: str) -> str:
    """
    GPT 응답에서 ```python 또는 ``` 등의 마크다운 코드 블럭을 제거
    """
    lines = code.strip().splitlines()

    # 앞뒤에 ```로 둘러싸인 경우 제거
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().endswith("```"):
        lines = lines[:-1]

    return "\n".join(lines)



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
            Assume that 'math' is already imported for you. Do NOT use any import statements.
            Your goal is to output only the function definition (no explanations, no markdown, no variable assignment).
            The function name MUST be 'f'.
            The function must take zero arguments.
            Do NOT call the function.
            Do NOT assign the result to a variable.
            0,1 and Negative number is not a Prime.
            Do NOT use 'eval', 'exec', 'os', '__', or any unsafe functions.
            You MUST NOT assign the result to a variable inside the function.

            Example:
            def f():
                return (2 + 7)
            """}
        ]
        code = get_chatbot_response(system_prompt_math + messages)
        code = clean_code_block(code)
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
