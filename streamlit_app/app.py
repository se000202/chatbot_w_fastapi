import streamlit as st
import requests
import os
st.write(f"Session messages count: {len(st.session_state.messages)}")
API_URL = os.getenv("FASTAPI_URL", "https://web-production-b2180.up.railway.app/chat")
st.write(f"API_URL = {API_URL}")  # 여기 추가 → URL 확인용

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

st.title("🗨️ Chatbot with Context (FastAPI + GPT)")

for msg in st.session_state.messages:
    if msg["role"] != "system":
        if msg["role"] == "user":
            st.write(f"🧑‍💻 **You:** {msg['content']}")
        elif msg["role"] == "assistant":
            st.write(f"🤖 **Bot:** {msg['content']}")

user_input = st.text_input("Your message:", "")

if st.button("Send"):
    if user_input.strip() != "":
        st.session_state.messages.append({"role": "user", "content": user_input})

        st.write("Sending POST request...")  # 디버그용 출력
        st.write(st.session_state.messages)  # 현재 messages 확인

        try:
            response = requests.post(
                API_URL,
                json={"messages": st.session_state.messages}
            )
            st.write(f"Response status code: {response.status_code}")  # 응답 코드 출력
            st.write(f"Response body: {response.text}")  # 응답 본문 출력

            if response.status_code == 200:
                bot_reply = response.json()["response"]
                st.session_state.messages.append({"role": "assistant", "content": bot_reply})
                st.experimental_rerun()
            else:
                st.error(f"Error {response.status_code}: {response.text}")

        except Exception as e:
            st.error(f"Exception: {str(e)}")

if st.button("Clear Chat"):
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    st.experimental_rerun()
