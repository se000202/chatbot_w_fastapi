import openai
import os
from dotenv import load_dotenv
from typing import List, Dict

# Load .env (로컬 실행 시 사용)
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_chatbot_response(messages: List[Dict[str, str]]) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    return response["choices"][0]["message"]["content"].strip()
