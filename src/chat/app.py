import os
import sys


import torch
from langchain_core.messages import trim_messages

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
    model_id = "Qwen/Qwen2-0.5B"
    model, tokenizer, pipe = load_hf_model(model_id, device)

    llm = create_pipeline(pipe)
    prompt = get_chat_with_history_template()
    trimmer = trim_messages(
        max_tokens=100,
        strategy="last",
        token_counter=llm,
        include_system=False,
        start_on="system",

    )
    parser = AIOutputParser()
    ch = prompt | trimmer

    config = {"ability": "medical", "input": "Hi who are you"},
    cc = prompt.invoke(config)

    chatbot_ = create_chatbot(prompt | llm | parser)

    return chatbot_


chatbot = init_chatbot()


@cl.on_chat_start
def on_chat_start():
    session_id = generate_session_id()
    cl.user_session.set("session_id", session_id)
    print(f"session_id: {session_id}")


@cl.on_message
async def on_message(message: cl.Message):
    session_id = cl.user_session.get("session_id")
    config = {"configurable": {"session_id": session_id}}

    output = chatbot.invoke(
        {"ability": "medical", "input": message.content},
        config=config
    )
    await cl.Message(content=f"{output}").send()


if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)