import openai
import os
from dotenv import load_dotenv

# Load .env (로컬 실행 시 사용)
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_chatbot_response(user_message: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": user_message}
        ]
    )
    return response["choices"][0]["message"]["content"].strip()
