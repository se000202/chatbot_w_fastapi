from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import os
from dotenv import load_dotenv
from openai import OpenAI
import re
from math import prod

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

# Expression calculation (safe eval)
def compute_expression(expr: str) -> str:
    try:
        # Provide math functions and built-ins safely
        safe_globals = {
            "__builtins__": {},
            "sum": sum,
            "range": range,
            "prod": prod,
            "round": round,
            "reduce": __import__('functools').reduce,
            "all": all
        }
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
                                          "You must use list comprehension with 'all(x % d != 0 ...)' inline to detect primes. "
                                          "If the user asks for sum of primes, output 'sum([...])' with inline prime detection. "
                                          "If the user asks for product of primes, output 'prod([...])' with inline prime detection. "
                                          "If the user asks for the nth Fibonacci number, output a one-line expression using reduce or Binet's formula. "
                                          "Only output the expression and nothing else."},
            {"role": "user", "content": last_msg}
        ]
        expr = get_chatbot_response(system_prompt)
        print(f"[DEBUG] Generated expression: {expr}")

        result = compute_expression(expr)
        return {"response": result}

    # Default chat
    answer = get_chatbot_response(messages)
    return {"response": answer}
