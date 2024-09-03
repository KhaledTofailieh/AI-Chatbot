import re

from langchain_core.output_parsers import BaseOutputParser


class AIOutputParser(BaseOutputParser):
    def parse(self, text: str) -> str:
        ai_output = re.search(r'(.*)AI:(.*)', text, re.DOTALL)
        if ai_output:
            return ai_output.group(2).strip()
        else:
            return text.strip()

    @property
    def _type(self) -> str:
        return "ai_output_parser"
