from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import subprocess
import os
import re
from dotenv import load_dotenv
from openai import OpenAI

# Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# In-memory recent code storage (simple version, can be extended per session)
recent_code = {"last_code": ""}

# Request model
class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]

# Run Python code safely
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

# Extract latest code block from GPT response
def extract_code_block(text: str) -> str:
    pattern = r"```python(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return ""

# Simple heuristic to check if it's probably Python code
def is_probable_python_code(text: str) -> bool:
    keywords = ['def ', 'print(', 'for ', 'while ', 'if ', 'import ', 'class ', 'lambda ', '=']
    return any(kw in text for kw in keywords) or text.strip().startswith('```python')

# Detect intent
def detect_intent(messages: List[Dict[str, str]]) -> str:
    user_msgs = [m["content"] for m in messages if m["role"] == "user"]
    last_msg = user_msgs[-1] if user_msgs else ""

    prompt = (
        "Classify the user intent into one of these categories only: "
        "'write_code', 'run_code', 'chat'.\n"
        "Run_code should be used only if the user provides valid Python code to execute "
        "or explicitly says to run the previously generated code.\n"
        "If the user asks something like 'can you run the code above?', classify it as 'run_code'.\n"
        "If the user provides general chat or instructions not containing code, classify it as 'chat'.\n"
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

# Get GPT response
def get_chatbot_response(messages: List[Dict[str, str]]) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    return response.choices[0].message.content.strip()

# Main chat endpoint
@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    messages = req.messages

    # Detect intent
    intent = detect_intent(messages)

    # Intent handling
    if intent == "write_code":
        # Code generation
        code_response = get_chatbot_response(messages)

        # Extract code block and store it
        code_block = extract_code_block(code_response)
        recent_code["last_code"] = code_block

        return {"intent": intent, "code": code_response}

    elif intent == "run_code":
        user_msgs = [m["content"] for m in messages if m["role"] == "user"]
        code_to_run = user_msgs[-1] if user_msgs else ""

        # If user said "run previous code", use recent code block
        if code_to_run.strip().lower() in [
            "위 코드를 실행해 줘",
            "위 코드 실행해줘",
            "execute the above code",
            "run the previous code"
        ]:
            code_to_run = recent_code.get("last_code", "")

        # Try running code
        try:
            if not is_probable_python_code(code_to_run):
                raise ValueError("No valid Python code detected.")

            output = run_python_code(code_to_run)

            # Execution successful → GPT explain
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

        except Exception as e:
            # Fallback to chat if execution failed
            fallback_messages = messages.copy()
            fallback_messages.append({"role": "system", "content": "Previous code execution failed or was not valid. Please continue the conversation."})

            fallback_answer = get_chatbot_response(fallback_messages)

            return {
                "intent": "chat_fallback",
                "error": str(e),
                "response": fallback_answer
            }

    else:
        # General chat
        answer = get_chatbot_response(messages)
        return {"intent": intent, "response": answer}
