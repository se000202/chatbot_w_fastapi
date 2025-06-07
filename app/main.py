from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import os
from dotenv import load_dotenv
from openai import OpenAI
from math import prod
from functools import reduce
import math

# Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# Request model
class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]

# GPT chat response
def get_chatbot_response(messages: List[Dict[str, str]]) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    return response.choices[0].message.content.strip()

# Expression calculation (safe eval + forbidden keyword check)
def compute_expression(expr: str) -> str:
    try:
        # First, check forbidden keywords
        forbidden_keywords = ["import", "def", "exec", "eval", "os.", "__"]

        for keyword in forbidden_keywords:
            if keyword in expr:
                return f"계산 중 오류 발생: 금지된 표현식 '{keyword}' 이 감지되었습니다. 실행 중단."

        # Safe globals with minimal necessary functions
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

        print(f"[DEBUG] About to eval expression: {repr(expr)} (type: {type(expr)})")

        # Ensure expr is string
        if not isinstance(expr, str):
            raise ValueError("Expression is not a string!")

        result = eval(expr, safe_globals)
        return f"계산 결과: {result}"
    except Exception as e:
        return f"계산 중 오류 발생: {e}"

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    messages = req.messages

    # Extract last user message
    user_msgs = [m["content"] for m in messages if m["role"] == "user"]
    last_msg = user_msgs[-1] if user_msgs else ""

    # Trigger keywords
    calc_keywords = ["합", "곱", "피보나치", "product of primes", "sum of primes", "fibonacci"]

    # If message contains calculation keyword → use calculation mode
    if any(keyword in last_msg for keyword in calc_keywords):
        # Strong system prompt to avoid is_prime and enforce list comprehension
        system_prompt = [
            {"role": "system", "content": "You are an assistant that converts calculation requests into ONE-LINE Python expressions. "
                                          "You must NOT define functions. You must NOT use 'is_prime' or any undefined functions. "
                                          "You must NOT use import statements. You must NOT use eval or exec or os. "
                                          "You must use list comprehension with 'all(x % d != 0 ...)' inline to detect primes. "
                                          "If the user asks for sum of primes, output 'sum([x for x in range(2, N+1) if all(x % d != 0 for d in range(2, int(x**0.5)+1))])'. "
                                          "You must use range(2, N+1). Do NOT use range(1, N+1). "
                                          "If the user asks for product of primes, output 'prod([x for x in range(2, N+1) if all(x % d != 0 for d in range(2, int(x**0.5)+1))])'. "
                                          "You must use range(2, N+1). Do NOT use range(1, N+1). "
                                          "If the user asks for the nth Fibonacci number, you MUST use a one-line expression with 'reduce' only. "
                                          "Do NOT use Binet’s formula or lists. Only use reduce-based expression. "
                                          "Only output the expression and nothing else."},
            {"role": "user", "content": last_msg}
        ]
        expr = get_chatbot_response(system_prompt)
        print(f"[DEBUG] Generated expression: {repr(expr)}")  # Added repr for full debug

        result = compute_expression(expr)
        return {"response": result}

    # Default chat
    answer = get_chatbot_response(messages)
    return {"response": answer}
