from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ChatRequest(BaseModel):
    message: str

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

def detect_intent(user_message: str) -> str:
    prompt = (
        "Classify the user intent into one of these categories only: 'write_code', 'run_code', 'chat'.\n"
        "User message: " + user_message
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    intent = response.choices[0].message.content.strip().lower()
    # 간단히 의도 텍스트가 포함되어 있는지 확인
    if "write_code" in intent:
        return "write_code"
    elif "run_code" in intent:
        return "run_code"
    else:
        return "chat"

def get_chatbot_response(messages) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    return response.choices[0].message.content.strip()

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    user_message = req.message

    intent = detect_intent(user_message)

    if intent == "write_code":
        # GPT에게 코딩 요청 전달 → 코드 텍스트 생성
        messages = [
            {"role": "system", "content": "You are a helpful assistant who writes Python code."},
            {"role": "user", "content": user_message},
        ]
        code_response = get_chatbot_response(messages)
        return {"intent": intent, "code": code_response}

    elif intent == "run_code":
        # Python 코드 실행
        output = run_python_code(user_message)

        messages = [
            {"role": "system", "content": "You are a helpful assistant that explains Python code output."},
            {
                "role": "user",
                "content": (
                    f"I ran this Python code:\n```python\n{user_message}\n```\n"
                    f"The output was:\n```\n{output}\n```\n"
                    "Please explain the output."
                ),
            },
        ]
        answer = get_chatbot_response(messages)

        return {"intent": intent, "output": output, "response": answer}

    else:
        # 일반 대화
        messages = [
            {"role": "system", "content": "You are a friendly conversational assistant."},
            {"role": "user", "content": user_message},
        ]
        answer = get_chatbot_response(messages)

        return {"intent": intent, "response": answer}
