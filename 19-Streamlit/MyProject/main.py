import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import load_prompt
import glob
from dotenv import load_dotenv

# API key 로드
load_dotenv()

st.title("ChatGPT")

# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요")

# 처음 1번만 실행되는 코드
if "messages" not in st.session_state:
    # 대화 기록 저장
    st.session_state["messages"] = []

# 사이드바 생성
with st.sidebar:
    # 초기화 버튼 생성
    clear_btn = st.button("대화 초기화")

    prompt_files = glob.glob("prompts/*.yaml")
    selected_prompt = st.selectbox("프롬포트를 선택해 주세요", prompt_files, index=0)
    task_input = st.text_input("TASK 입력", "")


# 이전 대화 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 새 대화 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


def create_chain(prompt_filepath, task=""):
    # 프롬포트 적용
    prompt = load_prompt(prompt_filepath, encoding="utf-8")
    if task:
        prompt = prompt.partial(task=task)

    # 모델
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
    # 출력 파서
    output_parser = StrOutputParser()
    # 체인
    chain = prompt | llm | output_parser
    return chain


# 초기화 버튼 눌리면
if clear_btn:
    st.session_state["messages"] = []

print_messages()

# 만약 사용자의 입력이 들어오면
if user_input:
    # 웹에 대화 출력
    st.chat_message("user").write(user_input)
    # 체인 생성
    chain = create_chain(selected_prompt, task=task_input)

    # 스트릐밍 호출
    response = chain.stream({"question": user_input})
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
