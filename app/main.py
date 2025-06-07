from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# Request model
class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]

# Get GPT response
def get_chatbot_response(messages: List[Dict[str, str]]) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    return response.choices[0].message.content.strip()

# Main chat endpoint → Chat only!
@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    messages = req.messages
    answer = get_chatbot_response(messages)
    return {"response": answer}
