import torch

from chat.chatbot import create_chatbot
from model.model_loader import load_hf_model
from pipeline.pipeline import create_pipeline
from prompt.templates import get_chat_with_history_template


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model_id = "openai-community/gpt2"
    # model_id = "microsoft/biogpt"

    model_id = "Qwen/Qwen2-0.5B"
    model, tokenizer, pipe = load_hf_model(model_id, device)
    llm = create_pipeline(pipe)
    prompt = get_chat_with_history_template()

    chatbot = create_chatbot(prompt | llm)

    config = {"configurable": {"session_id": "session_id_example"}}
    while True:
        input_ = input("Enter Your Message.")
        output = chatbot.invoke(
            {"ability": "medical", "input": input_},
            config=config
        )
        print(output)


if __name__ == "__main__":
    main()
