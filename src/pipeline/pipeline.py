from langchain_huggingface import HuggingFacePipeline


def create_pipeline(pipe):
    return HuggingFacePipeline(pipeline=pipe)
