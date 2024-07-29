import os

from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from rater_metric_calculator.llm.prompts.input_template_placeholder import (
    input_template_placeholder,
)
from rater_metric_calculator.llm.prompts.basic_prompt import basic_prompt

ANTHROPIC_MODEL = "claude-3-5-sonnet-20240620"
TEMPERATURE = 0.7


def get_prompt(prompt_file: str) -> str:
    """
    Retrieves the prompt template based on the given prompt file.

    Args:
        prompt_file (str): Path to the prompt file.

    Returns:
        PromptTemplate: A LangChain PromptTemplate object containing the prompt.
    """
    if prompt_file:
        with open(prompt_file, "r") as file:
            prompt = file.read()
    else:
        prompt = basic_prompt

    return PromptTemplate.from_template(template=(prompt + input_template_placeholder))


def analyze_with_llm(report: str, prompt_file: str) -> str:
    """
    Analyzes a report using a language model.

    Args:
        report (str): The report to be analyzed.
        prompt_file (str): Path to the prompt file.

    Returns:
        str: The analysis result from the language model.
    """
    anthropic = ChatAnthropic(
        model=ANTHROPIC_MODEL,
        temperature=TEMPERATURE,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )

    llm = get_prompt(prompt_file) | anthropic | StrOutputParser()

    return llm.invoke({"input": report}).strip()
