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


def get_prompt(prompt_file: str) -> PromptTemplate:
    """
    Retrieves the prompt template based on the given prompt file.

    Args:
        prompt_file (str): Path to the prompt file.

    Returns:
        PromptTemplate: A LangChain PromptTemplate object containing the prompt.

    Raises:
        FileNotFoundError: If the specified prompt file is not found.
        IOError: If there's an error reading the prompt file.
        ValueError: If the prompt file is empty or invalid.
    """
    try:
        if prompt_file:
            with open(prompt_file, "r") as file:
                prompt = file.read()
                if not prompt.strip():
                    raise ValueError(
                        "The prompt file is empty or contains only whitespace."
                    )
        else:
            prompt = basic_prompt

        return PromptTemplate.from_template(prompt + input_template_placeholder)
    except FileNotFoundError:
        raise FileNotFoundError(f"The prompt file '{prompt_file}' was not found.")
    except IOError as e:
        raise IOError(f"Error reading the prompt file: {str(e)}")
    except Exception as e:
        raise ValueError(
            f"An unexpected error occurred while processing the prompt: {str(e)}"
        )


def analyze_with_llm(report: str, prompt_file: str = None) -> str:
    """
    Analyzes a report using a language model.

    Args:
        report (str): The report to be analyzed.
        prompt_file (str): Path to the prompt file.

    Returns:
        str: The analysis result from the language model.
    """
    try:
        anthropic = ChatAnthropic(
            model=ANTHROPIC_MODEL,
            temperature=TEMPERATURE,
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )

        llm = get_prompt(prompt_file) | anthropic | StrOutputParser()

        return llm.invoke({"input": report}).strip()
    except ValueError as e:
        raise ValueError(f"Error in prompt or model configuration: {str(e)}")
    except KeyError as e:
        raise KeyError(f"Missing environment variable or configuration key: {str(e)}")
    except Exception as e:
        raise RuntimeError(
            f"An unexpected error occurred during LLM analysis: {str(e)}"
        )
