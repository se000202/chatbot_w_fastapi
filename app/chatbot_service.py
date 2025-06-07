import openai
import os
from dotenv import load_dotenv
from typing import List, Dict

# Load .env (로컬 개발 시 사용)
load_dotenv()

# 최신 openai 라이브러리 (1.x 이상)에서는 client 객체 사용
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_chatbot_response(messages: List[Dict[str, str]]) -> str:
    print("Sending messages to OpenAI:", messages)  # 디버그용 로그
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        print("Received response:", response)  # 디버그용 로그
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("OpenAI API call failed:", str(e))
        raise
