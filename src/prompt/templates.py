from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder, PromptTemplate


def get_basic_template():
    template = "Write me a small article about {topic}."
    return PromptTemplate(input_variables=["topic"], template=template)


def get_chat_template():
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="Write an article related to the input topic in one paragraph about 25 words."
            ),
            HumanMessagePromptTemplate.from_template("{topic}"),
        ]
    )
    return prompt


def get_chat_with_history_template():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You're a clinical assistant who will help the clinician in ACCUMED. Your name is ACCUMED Assistant. ACCUMED is company that porvide RCM (Revenue cycle mamngement) sevices and it's based in UAE it's CTO name is Mohammed Gamal. we have a Backend leader named Bassem Adas, and Backend Developer named Nour Rubeh she is working on lookup service project.the front end lead is Khaled Omar he mainly focuced on unified gitway.  Keep your response clear with no more than 50 words. Keep your answers short.",
            ),
            MessagesPlaceholder(variable_name="message_history"),
            ("user", "{input}"),
            ("ai", "")
        ]
    )
    return prompt
