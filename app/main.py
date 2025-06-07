from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import subprocess
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]  # messages 리스트로 변경

def run_python_code(code: str) -> str:
    try:
        result = subprocess.run(
            ["python3", "-c", code],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return f"Error:\n{result.stderr.strip()}"
    except Exception as e:
        return f"Exception: {str(e)}"

def detect_intent(messages: List[Dict[str, str]]) -> str:
    # user 역할 메시지 중 마지막 메시지를 가져와 의도 판단
    user_msgs = [m["content"] for m in messages if m["role"] == "user"]
    last_msg = user_msgs[-1] if user_msgs else ""

    prompt = (
        "Classify the user intent into one of these categories only: 'write_code', 'run_code', 'chat'.\n"
        "User message: " + last_msg
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    intent = response.choices[0].message.content.strip().lower()
    if "write_code" in intent:
        return "write_code"
    elif "run_code" in intent:
        return "run_code"
    else:
        return "chat"

def get_chatbot_response(messages: List[Dict[str, str]]) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    return response.choices[0].message.content.strip()

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    messages = req.messages

    intent = detect_intent(messages)

    if intent == "write_code":
        # 코딩 요청일 때 GPT에게 그대로 전달
        code_response = get_chatbot_response(messages)
        return {"intent": intent, "code": code_response}

    elif intent == "run_code":
        # 실행 요청일 때, user 메시지 중 마지막을 코드로 간주
        user_msgs = [m["content"] for m in messages if m["role"] == "user"]
        code_to_run = user_msgs[-1] if user_msgs else ""

        output = run_python_code(code_to_run)

        explain_messages = [
            {"role": "system", "content": "You are a helpful assistant that explains Python code output."},
            {
                "role": "user",
                "content": (
                    f"I ran this Python code:\n```python\n{code_to_run}\n```\n"
                    f"The output was:\n```\n{output}\n```\n"
                    "Please explain the output."
                )
            }
        ]
        answer = get_chatbot_response(explain_messages)
        return {"intent": intent, "output": output, "response": answer}

    else:
        # 일반 대화
        answer = get_chatbot_response(messages)
        return {"intent": intent, "response": answer}
