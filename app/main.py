from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.models import ChatRequest, ChatResponse
from app.chatbot_service import get_chatbot_response

app = FastAPI()

# CORS 설정 (처음에는 * 허용 → 배포 후에는 도메인 제한 추천)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발 시에는 * 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Chatbot API with context is running"}

@app.post("/chat")
async def chat_endpoint(payload: ChatRequest):
    try:
        # 예시 OpenAI 호출 → 여기에 실제 호출 넣기 가능
        # bot_reply = call_openai_api(payload.messages)
        bot_reply = "Hello! How can I assist you today?"

        # ✅ 반드시 JSON으로 정상 응답
        return {"response": bot_reply}

    except Exception as e:
        # ✅ 반드시 JSON으로 에러 응답
        return JSONResponse(
            status_code=500,
            content={"error": f"Internal server error: {str(e)}"}
        )
