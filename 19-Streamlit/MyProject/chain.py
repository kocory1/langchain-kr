from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import load_prompt
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_teddynote.models import MultiModal


# 체인 생성
def create_chain(retriever, model_name="ollama"):
    if model_name == "ollama":
        # 프롬포트 적용
        # 단계 6: 프롬프트 생성(Create Prompt)
        # 프롬프트를 생성합니다.
        prompt = load_prompt("prompts/pdf-rag.yaml", encoding="utf-8")

        # 단계 7: 언어모델(LLM) 생성
        # 모델(LLM) 을 생성합니다.
        llm = ChatOllama(model="EEVE-Korean-10.8B:latest", temperature=0)
    else:
        prompt = load_prompt("prompts/pdf-rag-ollama.yaml", encoding="utf-8")
        llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0)
    # 단계 8: 체인(Chain) 생성

    chain = (
        {"context": retriever | format_doc, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def generate_answer(image_filepath, system_prompt, user_prompt, model="gpt-4o"):
    # 객체 생성
    llm = ChatOpenAI(
        temperature=0,  # 창의성 (0.0 ~ 2.0)
        model=model,  # 모델명
    )

    system_prompt = system_prompt
    user_prompt = user_prompt

    # 멀티모달 객체 생성
    multimodal = MultiModal(llm, system_prompt=system_prompt, user_prompt=user_prompt)
    answer = multimodal.stream(image_filepath)
    return answer


def format_doc(document_list):
    return "\n\n".join([doc.page_content for doc in document_list])
