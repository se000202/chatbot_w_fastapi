from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import openai
import os
from dotenv import load_dotenv

# .env 로드
load_dotenv()

# OpenAI API KEY 설정
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# Pydantic 모델 정의
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

@app.post("/chat")
async def chat_endpoint(payload: ChatRequest):
    try:
        # OpenAI API 호출 (GPT-4.1: gpt-4o 사용)
        response = openai.ChatCompletion.create(
            model="gpt-4o",   # 최신 GPT-4.1 모델
            messages=[{"role": msg.role, "content": msg.content} for msg in payload.messages],
            max_tokens=200,
            temperature=0.7
        )

        bot_reply = response["choices"][0]["message"]["content"].strip()

        # 정상 응답 반환
        return {"response": bot_reply}

    except Exception as e:
        # 에러 발생 시 JSON 응답 반환
        return JSONResponse(
            status_code=500,
            content={"error": f"Internal server error: {str(e)}"}
        )
