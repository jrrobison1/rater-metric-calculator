import os

from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser

from rater_metric_calculator.llm.prompts.basic_prompt import basic_prompt

ANTHROPIC_MODEL = "claude-3-5-sonnet-20240620"
TEMPERATURE = 0.5


def analyze_with_llm(report: str) -> str:
    anthropic = ChatAnthropic(
        model=ANTHROPIC_MODEL,
        temperature=TEMPERATURE,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )
    llm = basic_prompt | anthropic | StrOutputParser()

    return llm.invoke({"input": report}).strip()
