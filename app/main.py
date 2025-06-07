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
    return {"message": "Chatbot API is running"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    bot_reply = get_chatbot_response(request.message)
    return ChatResponse(response=bot_reply)
