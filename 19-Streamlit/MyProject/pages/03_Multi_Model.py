import os

import streamlit as st
from chain import generate_answer
from dotenv import load_dotenv
from langchain_core.messages.chat import ChatMessage
from langchain_teddynote import logging

# 프로젝트 이름을 입력합니다.s
logging.langsmith("[Project] MultiModal")

# API key 로드
load_dotenv()

# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")
if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")

# 파일 업로드 전용 폴더
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

st.title("이미지 인식 기반 챗봇")

# 처음 1번만 실행되는 코드
if "messages" not in st.session_state:
    # 대화 기록 저장
    st.session_state["messages"] = []

if "chain" not in st.session_state:
    # 아무런 파일을 업로드하지 않았을 경우
    st.session_state["chain"] = None

# 사이드바 생성
with st.sidebar:
    # 초기화 버튼 생성
    clear_btn = st.button("대화 초기화")
    # 이미지 업로드
    uploaded_file = st.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png"])
    # 모델 선택 메뉴
    selected_model = st.selectbox("LLM 선택", ["gpt-4o", "gpt-4o-mini"], index=0)
    # 시스템 프롬포트
    system_prompt = st.text_area("시스템 프롬포트", "", height=200)


# 이전 대화 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 새 메시지 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 이미지을 캐시 저장 (시간이 오래 걸리는 작업 처리 예정)
@st.cache_resource(show_spinner="업로드한 이미지를 처리 중입니다...")
def process_imagefile(file):
    # 업로드한 파일을 캐시 디렉토리에 저장합니다.
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    return file_path


# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요")

# 경고 메시지 영역
warning_message = st.empty()

# 초기화 버튼 눌리면
if clear_btn:
    st.session_state["messages"] = []

print_messages()

# 만약 사용자의 입력이 들어오면
if user_input:
    # 파일이 업로드되었는지 확인
    if uploaded_file:
        image_filepath = process_imagefile(uploaded_file)
        st.image(image_filepath)
        # 답변 요청
        response = generate_answer(
            image_filepath=image_filepath,
            system_prompt=system_prompt,
            user_prompt=user_input,
            model=selected_model,
        )

        st.chat_message("user").write(user_input)

        with st.chat_message("assistant"):
            # 빈 컨테이너 생성하여 토큰을 스트리밍 출력
            container = st.empty()

            ai_answer = ""
            for token in response:
                ai_answer += token.content
                container.markdown(ai_answer)

        # 대화 기록 저장
        add_message("user", user_input)
        add_message("assistant", ai_answer)
    else:
        # 이미지 업로드 X -> chain이 None이므로 경고문 출력
        warning_message.error("이미지를 업로드하지 않았습니다")
