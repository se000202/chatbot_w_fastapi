# ✅ app.py — Streamlit 최종 개선본 (Send → /chat, Streaming → /chat_stream, 줄바꿈 + 이모지 + code mode 표시)

import streamlit as st
import requests
import os
from dotenv import load_dotenv

# Load env
load_dotenv()

# API URL 설정
API_URL = os.getenv("FASTAPI_URL")
if not API_URL:
    st.error("❌ API_URL is not set! Please check your environment variables.")
    st.stop()

# messages 초기화
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

# user_input_key_num 및 user_input_key 초기화
if "user_input_key_num" not in st.session_state:
    st.session_state.user_input_key_num = 0
if "user_input_key" not in st.session_state:
    st.session_state.user_input_key = f"user_input_{st.session_state.user_input_key_num}"

# last_is_code 초기화
if "last_is_code" not in st.session_state:
    st.session_state["last_is_code"] = False

# UI 구성
st.title("💬 Chatbot with Streaming + Context (FastAPI + GPT)")

# reply_box 전역 선언
reply_box = st.empty()

# code 키워드 설정
code_keywords = ["python 코드", "파이썬 코드", "python function", "python program"]

# 이전 대화 표시
for i, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        st.write(f"🧑‍💼 **You:** {msg['content']}")

        # 현재 user message 가 code mode 인지 표시 flag 저장
        st.session_state["last_is_code"] = any(keyword in msg["content"] for keyword in code_keywords)

    elif msg["role"] == "assistant":
        safe_content = msg["content"]

        # Bot prefix 결정
        if st.session_state.get("last_is_code", False):
            bot_prefix = "🤖 **Bot (code mode):**"
        else:
            bot_prefix = "🤖 **Bot:**"

        # 출력
        if i == len(st.session_state.messages) - 1 and st.session_state.get("streaming", False):
            reply_box.markdown(f"{bot_prefix} {safe_content}", unsafe_allow_html=False)
        else:
            st.markdown(f"{bot_prefix} {safe_content}", unsafe_allow_html=False)

# 사용자 입력
user_input = st.text_area("Your message:", height=100, key=st.session_state.user_input_key)

# Send 버튼
if st.button("Send"):
    user_input_value = st.session_state.get(st.session_state.user_input_key, "").strip()

    if user_input_value != "":
        st.session_state.messages.append({
            "role": "user",
            "content": user_input_value
        })

        st.session_state.user_input_key_num += 1
        st.session_state.user_input_key = f"user_input_{st.session_state.user_input_key_num}"

        st.session_state.messages.append({
            "role": "assistant",
            "content": ""
        })
        st.session_state.streaming = True

        with st.spinner("Assistant is responding..."):
            response = requests.post(
                API_URL + "/chat",  # ✅ /chat endpoint 호출 (stream 제거)
                json={"messages": st.session_state.messages}
            )

            if response.status_code == 200:
                try:
                    resp_json = response.json()
                    if "response" in resp_json:
                        st.session_state.messages[-1]["content"] = resp_json["response"]
                    else:
                        st.session_state.messages[-1]["content"] = f"❌ Invalid response format: {resp_json}"
                except Exception as e:
                    st.session_state.messages[-1]["content"] = f"❌ Error parsing JSON: {str(e)}\nResponse text: {response.text}"
            else:
                st.session_state.messages[-1]["content"] = f"❌ Error {response.status_code}: {response.text}"

            # Bot prefix 결정
            if st.session_state.get("last_is_code", False):
                bot_prefix = "🤖 **Bot (code mode):**"
            else:
                bot_prefix = "🤖 **Bot:**"

            reply_box.markdown(f"{bot_prefix} {st.session_state.messages[-1]['content']}", unsafe_allow_html=False)

        st.session_state.streaming = False
        st.rerun()


# Clear Chat 버튼
if st.button("Clear Chat"):
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    st.session_state.user_input_key_num += 1
    st.session_state.user_input_key = f"user_input_{st.session_state.user_input_key_num}"
    st.session_state["last_is_code"] = False  # 리셋
    st.rerun()
