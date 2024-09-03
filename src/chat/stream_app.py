import os
import sys

import torch
from langchain_core.messages import trim_messages
import streamlit as st

# add project path to python path.
directory_path = os.path.dirname(os.path.dirname(__file__))
path_parts = directory_path.split(os.path.sep)
root_path = os.path.sep.join(path_parts[:-1])
sys.path.insert(0, root_path)

from src.chat.chatbot import create_chatbot
from src.model.model_loader import load_hf_model
from src.pipeline.pipeline import create_pipeline
from src.prompt.templates import get_chat_with_history_template
from src.utils.parsers import AIOutputParser
from src.utils.functions import generate_session_id


def init_chatbot():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "aaditya/Llama3-OpenBioLLM-8B" #"Qwen/Qwen2-0.5B" "Qwen/Qwen2-0.5B" # "internistai/base-7b-v0.2" # 
    model, tokenizer, pipe = load_hf_model(model_id, device)

    llm = create_pipeline(pipe)
    prompt = get_chat_with_history_template()
    # trimmer = trim_messages(
    #     max_tokens=100,
    #     strategy="last",
    #     token_counter=llm,
    #     include_system=False,
    #     start_on="system",

    # )
    parser = AIOutputParser()
    # ch = prompt | trimmer

    # config = {"ability": "medical", "input": "Hi who are you"},
    # cc = prompt.invoke(config)

    chatbot_ = create_chatbot(prompt | llm | parser)

    return chatbot_


chatbot = init_chatbot()


if "session_id" not in st.session_state:
    st.session_state["session_id"] = generate_session_id()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


user_message = st.chat_input("Ask Me A Question!")
if user_message:
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = generate_session_id()
 
    st.session_state.messages.append({"role": "user", "content": user_message})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_message)

    config = {"configurable": {"session_id": st.session_state["session_id"]}}

    # Display assistant response in chat message container
    # "ability": "icd coding", 
    with st.chat_message("assistant"):
        output = chatbot.invoke({"input": str(user_message)}, config=config)
        st.write(output)

    st.session_state.messages.append({"role": "assistant", "content": output})
