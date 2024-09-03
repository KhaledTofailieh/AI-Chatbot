from setuptools import setup, find_packages

setup(
    name="medical_chatbot",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "transformers",
        "langchain",
        "torch",
        "langchain-huggingface",
        "langchain-community",
        "pypdf",
        "chromadb",
        "chainlit",
        "qdrant-client",
        "bitsandbytes",
        "accelerate",
        "sacremoses"
    ],
    entry_points={
        "console_scripts": [
            "run_chatbot=src.main:main",
        ]
    },
)
