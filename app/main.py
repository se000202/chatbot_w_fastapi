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

# GPT Streaming generator
def gpt_stream(messages):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        stream=True
    )
    for chunk in response:
        if 'choices' in chunk and len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            if 'content' in delta:
                yield delta.content

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
    # 이미 $$가 있으면 그대로 둔다
    if "$$" in response:
        return response

    # LaTeX environments to capture and convert
    environments = [
        "align\\*", "align", "equation", "gather", "multline"
    ]

    # \[ ... \] → $$ ... $$ 변환
    response = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', response, flags=re.DOTALL)

    # \begin{ENV}...\end{ENV} → $$ ... $$ 변환
    for env in environments:
        pattern = rf'\\begin{{{env}}}(.*?)\\end{{{env}}}'
        response = re.sub(pattern, r'$$\1$$', response, flags=re.DOTALL)

    # 너무 긴 문장은 감싸지 않음 (설명 가능성 높음)
    if len(response.strip()) > 300:
        return response

    # 수식 keyword heuristic (안전망 유지)
    formula_keywords = ["=", "+", "-", "*", "/", "^", "\\approx","\\sqrt", "\\frac", "\\sum", "\\int", "\\log", "\\sin", "\\cos", "\\tan", "\\text", "\\displaystyle"]

    if any(keyword in response for keyword in formula_keywords):
        if "\n" not in response and all(c.isalnum() or c in " +-*/^=()[]{}\\._" for c in response.strip()):
            print("[DEBUG] Auto-wrapping response as LaTeX (fallback keyword match).")
            return f"$$ {response.strip()} $$"

    return response

# ✅ FastAPI @app.post("/chat") 수정된 endpoint 버전

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    messages = req.messages

    user_msgs = [m["content"] for m in messages if m["role"] == "user"]
    last_msg = user_msgs[-1] if user_msgs else ""

    calc_keywords = ["합", "곱", "피보나치", "product of primes", "sum of primes", "fibonacci"]

    if any(keyword in last_msg for keyword in calc_keywords):
        # 계산 모드 → 기존 방식 유지 (stream X)
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
        print(f"[DEBUG] Generated expression: {repr(expr)}")

        result = compute_expression(expr)
        return {"response": result}

    # 일반 chat 처리 → stream 파라미터 사용 여부 확인
    # req 객체에는 stream 파라미터 없음 → 클라이언트에서 messages 외에 stream 여부 명시 필요
    # 여기서는 임시로 stream 여부를 query param 으로 받는 예시로 작성
    # 실제 프로덕션에서는 별도의 /chat_stream endpoint 추천

    from fastapi import Request
    from fastapi.params import Query

    # stream 파라미터 기본 False
    import inspect
    frame = inspect.currentframe().f_back
    request = frame.f_locals.get("request", None)

    stream_mode = False
    if request is not None:
        try:
            stream_param = request.query_params.get("stream", "false")
            stream_mode = stream_param.lower() == "true"
        except:
            pass

    system_prompt_default = [
        {"role": "system", "content": "You are a helpful assistant. "
                                      "If your output includes a mathematical formula or expression, surround it with $$...$$ "
                                      "so that it can be rendered as LaTeX. "
                                      "If your output is normal text, do not use $$."},
    ]

    if stream_mode:
        # Streaming mode 사용
        return StreamingResponse(
            gpt_stream(system_prompt_default + messages),
            media_type="text/plain"
        )
    else:
        # 일반 JSON 응답
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=system_prompt_default + messages
        )
        return {"response": response.choices[0].message.content.strip()}

# 🚀 추후 확장 가능: /chat_stream 별도 endpoint 구성 추천!


