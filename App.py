import streamlit as st
import os
from Agent2 import Agent
import random
import math


st.set_page_config(page_title="Agentic RAG Assistant", layout="wide")
left_col, right_col = st.columns([3, 1])


@st.cache_resource
def return_model(model_type, text_split, text_overlap):
    model = Agent(model_type, st.secrets['openai_key'], st.secrets['deepseek_key'], st.secrets['langchain_key'])
    model.set_text_split(text_split, text_overlap)
    return model
    
    
tmp_dir = "storage/tmp_docs/"
mem_dir = "storage/mem_docs/"
if not os.path.isdir(tmp_dir):
    os.makedirs(tmp_dir, exist_ok=True)
if not os.path.isdir(mem_dir):
    os.makedirs(mem_dir, exist_ok=True)


with right_col:

    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = 1
    if 'disable_opt' not in st.session_state:
        st.session_state["disable_opt"] = False

    uploaded_files = st.file_uploader("Upload your files", accept_multiple_files=True, key=st.session_state["uploader_key"])
    model_choice = st.selectbox("Choose LLM", 
                                ["deepseek_reasoner", "deepseek_chat", "openai"],index=None,
                                placeholder="Select LLM",disabled = st.session_state["disable_opt"])
    
    if not st.session_state["disable_opt"] and model_choice:
        st.session_state["disable_opt"] = True
        
    if model_choice:
        text_split = int(math.ceil(st.number_input("Insert splitting size", value=10000)))
        text_overlap = int(math.ceil(st.number_input("Insert overlap size", value=2000)))
        model = return_model(model_choice, text_split, text_overlap)
        if st.button("Reset", type="primary"):
            model.reset(tmp_dir, mem_dir)
            st.cache_resource.clear()
            st.session_state["uploader_key"] += 1
            st.session_state["disable_opt"] = False
            if "last_response" in st.session_state:
                del st.session_state["last_response"]
            st.rerun()
        if st.button("Clear files", type = "primary"):
            model.clear_files(tmp_dir, mem_dir)
        if "last_response" in st.session_state:
                st.download_button(
                    label="Download Response",
                    data=st.session_state["last_response"],
                    file_name="response.md",
                    mime=None,
                    icon=":material/download:",
                    on_click="ignore"
                )


with left_col:
    st.title("Research Assistant")
    response = None

    if uploaded_files:
        for file in uploaded_files:
            with open(os.path.join(tmp_dir, file.name), "wb") as f:
                f.write(file.getbuffer())
                
    if model_choice:
        
        if "last_response" in st.session_state:
            with st.container(height=500):
                st.markdown("### Response")
                st.write(st.session_state["last_response"])
        
        question = st.chat_input("Enter your question...")
        
        if question:
            with st.spinner("Thinking..."):
                st.session_state["last_response"] = ""
                response = model.ask(question, tmp_dir, mem_dir)
            with st.container(height=500):
                with open(os.path.join(mem_dir, "response.md"), "w",encoding="utf-8") as f:
                    f.write(response)
            st.session_state["last_response"] = response
            # st.download_button(
            #     label="Download Response",
            #     data=response,
            #     file_name="response.md",
            #     mime=None,
            #     icon=":material/download:",
            #     on_click="ignore"
            # )
            st.rerun()