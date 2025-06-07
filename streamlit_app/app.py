import streamlit as st
import requests

# FastAPI 서버 URL
API_URL = "https://web-production-b2180.up.railway.app/chat"

# Session state에 messages 리스트 유지 (초기화)
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

st.title("🗨️ Chatbot with Context (FastAPI + GPT)")

# 이전 대화 내용 출력
for msg in st.session_state.messages:
    if msg["role"] != "system":
        if msg["role"] == "user":
            st.write(f"🧑‍💻 **You:** {msg['content']}")
        elif msg["role"] == "assistant":
            st.write(f"🤖 **Bot:** {msg['content']}")

# 사용자 입력 받기
user_input = st.text_input("Your message:", "")

# 전송 버튼
if st.button("Send"):
    if user_input.strip() != "":
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": user_input})

        try:
            # FastAPI 서버로 POST 요청
            response = requests.post(
                API_URL,
                json={"messages": st.session_state.messages}
            )
            if response.status_code == 200:
                bot_reply = response.json()["response"]
                # Assistant 응답 추가
                st.session_state.messages.append({"role": "assistant", "content": bot_reply})
                # 입력창 비우기 (재렌더링 위해 rerun)
                st.experimental_rerun()
            else:
                st.error(f"Error {response.status_code}: {response.text}")

        except Exception as e:
            st.error(f"Exception: {str(e)}")
