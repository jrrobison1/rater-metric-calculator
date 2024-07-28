from unittest.mock import MagicMock, patch

import pytest

from rater_metric_calculator.llm.analyze_with_llm import analyze_with_llm


@pytest.fixture
def mock_anthropic():
    with patch("rater_metric_calculator.llm.analyze_with_llm.ChatAnthropic") as mock:
        yield mock


@pytest.fixture
def mock_basic_prompt():
    with patch("rater_metric_calculator.llm.analyze_with_llm.basic_prompt") as mock:
        yield mock


def test_analyze_with_llm(mock_anthropic, mock_basic_prompt):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = "Mocked LLM response"
    mock_basic_prompt.__or__.return_value.__or__.return_value = mock_llm

    result = analyze_with_llm("Test report")

    assert result == "Mocked LLM response"
    mock_anthropic.assert_called_once_with(
        model="claude-3-5-sonnet-20240620",
        temperature=0.5,
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
    mock_anthropic, mock_basic_prompt, report, expected
):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = expected
    mock_basic_prompt.__or__.return_value.__or__.return_value = mock_llm

    result = analyze_with_llm(report)

    assert result == expected
    mock_llm.invoke.assert_called_once_with({"input": report})


def test_analyze_with_llm_error_handling(mock_anthropic, mock_basic_prompt):
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = Exception("API Error")
    mock_basic_prompt.__or__.return_value.__or__.return_value = mock_llm

    with pytest.raises(Exception, match="API Error"):
        analyze_with_llm("Test report")
