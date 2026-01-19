import os

import streamlit as st
from chain import create_chain
from dotenv import load_dotenv
from langchain_core.messages.chat import ChatMessage
from langchain_teddynote import logging
from retriever import create_retriever

# 프로젝트 이름을 입력합니다.s
logging.langsmith("[Project] PDF RAG")

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

st.title("Local Model 기반 RAG")

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
    # 파일 업로더
    uploaded_file = st.file_uploader("파일 업로드", type=["pdf"])
    # 모델 선택 메뉴
    selected_model = st.selectbox("LLM 선택", ["xionic", "ollama"], index=0)


# 이전 대화 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 새 메시지 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 파일을 캐시 저장 (시간이 오래 걸리는 작업 처리 예정)
@st.cache_resource(show_spinner="업로드한 파일을 처리 중입니다...")
def embed_file(file):
    # 업로드한 파일을 캐시 디렉토리에 저장합니다.
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    return create_retriever(file_path=file_path)


# 파일이 업로드 되었을 때
if uploaded_file:
    # 파일 업로드 후 retriever 생성
    retriever = embed_file(uploaded_file)
    chain = create_chain(retriever, model_name=selected_model)
    st.session_state["chain"] = chain

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
    # 체인 생성
    chain = st.session_state["chain"]

    # chain이 None이 아닐 경우(파일 업로드 했다면)
    if chain is not None:
        # 웹에 대화 출력
        st.chat_message("user").write(user_input)
        # 스트릐밍 호출
        response = chain.stream(user_input)
        with st.chat_message("assistant"):
            # 빈 컨테이너 생성하여 토큰을 스트리밍 출력
            container = st.empty()

            ai_answer = ""
            for token in response:
                ai_answer += token
                container.markdown(ai_answer)

        # 대화 기록 저장
        add_message("user", user_input)
        add_message("assistant", ai_answer)
    else:
        # 파일 업로드 X -> chain이 None이므로 경고문 출력
        warning_message.error("파일을 업로드하지 않았습니다")
