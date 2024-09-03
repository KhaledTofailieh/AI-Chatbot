from langchain_core.runnables.history import RunnableWithMessageHistory

from .history import get_session_history


def create_chatbot(runnable):
    with_message_history = RunnableWithMessageHistory(
        runnable,
        get_session_history,
        input_messages_key="input",
        history_messages_key="message_history",
    )
    return with_message_history
