from unittest.mock import MagicMock, patch, mock_open

import pytest

from langchain_core.prompts import PromptTemplate

from rater_metric_calculator.llm.analyze_with_llm import (
    analyze_with_llm,
    get_prompt,
    basic_prompt,
    input_template_placeholder,
)


@pytest.fixture
def mock_anthropic():
    with patch("rater_metric_calculator.llm.analyze_with_llm.ChatAnthropic") as mock:
        yield mock


@pytest.fixture
def mock_prompt_template():
    with patch("rater_metric_calculator.llm.analyze_with_llm.PromptTemplate") as mock:
        yield mock


def test_analyze_with_llm(mock_anthropic, mock_prompt_template):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = "Mocked LLM response"
    mock_prompt_template.from_template.return_value.__or__.return_value.__or__.return_value = (
        mock_llm
    )

    result = analyze_with_llm("Test report")

    assert result == "Mocked LLM response"
    mock_anthropic.assert_called_once_with(
        model="claude-3-5-sonnet-20240620",
        temperature=0.7,
        api_key=None,  # Since we're not setting the environment variable in the test
    )
    mock_llm.invoke.assert_called_once_with({"input": "Test report"})


@pytest.mark.parametrize(
    "report,expected",
    [
        ("", ""),
        ("Short report", "Short report response"),
        ("Long detailed report", "Long detailed response"),
    ],
)
def test_analyze_with_llm_various_inputs(
    mock_anthropic, mock_prompt_template, report, expected
):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = expected
    mock_prompt_template.from_template.return_value.__or__.return_value.__or__.return_value = (
        mock_llm
    )

    result = analyze_with_llm(report)

    assert result == expected
    mock_llm.invoke.assert_called_once_with({"input": report})


def test_analyze_with_llm_error_handling(mock_anthropic, mock_prompt_template):
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = Exception("API Error")
    mock_prompt_template.from_template.return_value.__or__.return_value.__or__.return_value = (
        mock_llm
    )

    with pytest.raises(Exception, match="API Error"):
        analyze_with_llm("Test report")


def test_get_prompt_with_file():
    custom_prompt = "Custom prompt content"
    mock_file = mock_open(read_data=custom_prompt)
    with patch("builtins.open", mock_file):
        result = get_prompt("custom_prompt.txt")

    assert isinstance(result, PromptTemplate)
    assert custom_prompt in result.template


def test_get_prompt_without_file(mock_prompt_template):
    mock_prompt_template.from_template.return_value = PromptTemplate.from_template(
        "Test template"
    )
    result = get_prompt(None)

    mock_prompt_template.from_template.assert_called_once_with(
        basic_prompt + input_template_placeholder
    )
    assert isinstance(result, PromptTemplate)
    assert result.template == "Test template"


def test_analyze_with_llm_custom_prompt(mock_anthropic, mock_prompt_template):
    custom_prompt = "Custom prompt content"
    mock_file = mock_open(read_data=custom_prompt)
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = "Custom prompt response"
    mock_prompt_template.from_template.return_value.__or__.return_value.__or__.return_value = (
        mock_llm
    )

    with patch("builtins.open", mock_file):
        result = analyze_with_llm("Test report", prompt_file="custom_prompt.txt")

    assert result == "Custom prompt response"
    mock_file.assert_called_once_with("custom_prompt.txt", "r")
    mock_llm.invoke.assert_called_once_with({"input": "Test report"})
