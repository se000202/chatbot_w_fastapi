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
import requests
from bs4 import BeautifulSoup
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
        sys.set_int_max_str_digits(100000)
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
        # safe_globals = {
        #     "__builtins__": {},
        #     "math": math,
        #     "sum": sum,
        #     "range": range,
        #     "prod": prod,
        #     "round": round,
        #     "reduce": reduce,
        #     "all": all,
        #     "int": int,
        #     "float": float,
        #     "abs": abs,
        #     "pow": pow,
        #     "len": len
        # }

        # local_vars = {}

        # 함수 정의 실행
        exec(code)

        # 함수 호출
        # if "f" not in local_vars:
        #     raise ValueError("함수 f 가 정의되어 있지 않습니다!")

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
    calc_keywords = [
        "합", "곱", "피보나치", "피보나치 수", "product of primes", "sum of primes",
        "fibonacci", "소수", "소수의 합", "소수의 곱", "prime",
        "표준편차", "분산", "평균", "median", "variance", "standard deviation"
    ]
    # Case 1 : 미국주식 사관학교가 input에 있을 떄
    if "미국주식 사관학교" in last_msg:
        try:
            url = "https://contents.premium.naver.com/usa/nasdaq/contents/250409170641442po"
            headers = {"User-Agent": "Mozilla/5.0"}
            res = requests.get(url, headers=headers)
            soup = BeautifulSoup(res.text, "html.parser")
            main_container = soup.select_one("div.se-main-container")
            texts = main_container.select("div.se-component.se-text.se-l-default")
            content = "\n\n".join(
                block.get_text(strip=True) for block in texts if block.get_text(strip=True)
            )
            system_prompt_summary = [
                {"role": "system", "content": "You are a summarization expert who summarizes long Korean articles into 3-5 bullet points."},
                {"role": "user", "content": content}
            ]
            summary = get_chatbot_response(system_prompt_summary)
            return {"response": summary}
        except Exception as e:
            return {"response": f"❌ 크롤링 중 오류 발생: {e}"}

    # Case 2: 수학/코드 관련 keywords
    elif any(keyword in last_msg for keyword in calc_keywords):
        # Case 1: Python 함수 정의 요청
        system_prompt_math = [
            {"role": "system", "content": """
            You are professional python programmer which does the two Following tasks.
            Task 1 : When the chat is given from the user, you should determine whether given chat can be converted to the task,
            which can be solved by using the python program.
            If the given chat can be converted into task, return the string which initial three character is "YES," and continued string is 
            the Task which the chat is converted.
            Otherwise, return "NO!,"
            
            Task 2 : You had recevied the string from Task 1. If the first four character is "NO!," return empty string. 
            Otherwise write the python program which solves the Task of the given string.
            The program must include the function named "main" and the must call the function "main".
            the main function must print the result of output which the task requires.
            Do NOT use 'eval', 'exec', 'os', '__', or any unsafe functions.
            You MUST NOT assign the result to a variable inside the function.
            
            Example:
            - Given Task : "YES, what is the sum from 1 to 10? 
            - Example Respone:
            def main(S,E):
                s = 0
                for i in range(S,E+1):
                    s += i
                print(s)
            main(1,10)
            Note : 0,1 and Negative number is not a Prime.
            """}
        ]
        code = get_chatbot_response(system_prompt_math + messages)
        if code:
            code = clean_code_block(code)
            result = safe_exec_function(code)
            return {"response": result}
        else:
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
