import streamlit as st
import requests
import os

# 반드시 /chat까지 포함
API_URL = os.getenv("FASTAPI_URL", "https://web-production-b2180.up.railway.app/chat")

# messages 초기화
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

st.title("🗨️ Chatbot with Context (FastAPI + GPT)")

# 이전 대화 표시
for msg in st.session_state.messages:
    if msg["role"] != "system":
        if msg["role"] == "user":
            st.write(f"🧑‍💻 **You:** {msg['content']}")
        elif msg["role"] == "assistant":
            st.write(f"🤖 **Bot:** {msg['content']}")

# 입력창
user_input = st.text_input("Your message:", "")

# Send 버튼
if st.button("Send"):
    if user_input.strip() != "":
        st.session_state.messages.append({"role": "user", "content": user_input})

        try:
            response = requests.post(
                API_URL,
                json={"messages": st.session_state.messages}
            )
            if response.status_code == 200:
                bot_reply = response.json()["response"]
                st.session_state.messages.append({"role": "assistant", "content": bot_reply})
                st.experimental_rerun()
            else:
                st.error(f"Error {response.status_code}: {response.text}")

        except Exception as e:
            st.error(f"Exception: {str(e)}")

# Clear Chat 버튼
if st.button("Clear Chat"):
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    st.experimental_rerun()
