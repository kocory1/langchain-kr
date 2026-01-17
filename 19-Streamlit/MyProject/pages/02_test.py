import streamlit as st

st.title("Test Upload")
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    # 파일을 읽지 않고 이름만 출력 (가장 가벼운 동작)
    st.write("Filename:", uploaded_file.name)
    st.success("Success!")
