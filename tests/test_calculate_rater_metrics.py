import pytest
import pandas as pd
import numpy as np
from rater_metric_calculator.calculate_rater_metrics import (
    calculate_krippendorff_alpha,
    calculate_fleiss_kappa,
    calculate_overall_percent_agreement,
    calculate_kendalls_w,
    calculate_mean_absolute_difference,
    calculate_rating_distribution,
    calculate_pairwise_metrics,
    calculate_pairwise_metrics_for_all,
    calculate_overall_metrics,
    PairwiseResult,
)
from unittest.mock import patch, mock_open
from rater_metric_calculator.calculate_rater_metrics import main


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "Rater1": [1, 2, 3, 4, 5],
            "Rater2": [1, 2, 3, 4, 5],
            "Rater3": [2, 2, 3, 4, 4],
        }
    )


def test_calculate_krippendorff_alpha(sample_data):
    alpha = calculate_krippendorff_alpha(sample_data)
    assert isinstance(alpha, float)
    assert 0 <= alpha <= 1


def test_calculate_fleiss_kappa(sample_data):
    kappa = calculate_fleiss_kappa(sample_data)
    assert isinstance(kappa, float)
    assert -1 <= kappa <= 1


def test_calculate_overall_percent_agreement(sample_data):
    agreement = calculate_overall_percent_agreement(sample_data)
    assert isinstance(agreement, float)
    assert 0 <= agreement <= 100


def test_calculate_kendalls_w(sample_data):
    w = calculate_kendalls_w(sample_data)
    assert isinstance(w, float)
    assert 0 <= w <= 1


def test_calculate_mean_absolute_difference(sample_data):
    mad = calculate_mean_absolute_difference(sample_data)
    assert isinstance(mad, float)
    assert 0 <= mad <= 4  # Max difference between 1 and 5


def test_calculate_rating_distribution(sample_data):
    dist = calculate_rating_distribution(sample_data)
    assert isinstance(dist, pd.DataFrame)
    assert dist.shape == (5, 3)  # 5 ratings, 3 raters
    assert np.allclose(dist.sum(), 100)  # Each column should sum to 100%


def test_calculate_pairwise_metrics(sample_data):
    result = calculate_pairwise_metrics(sample_data, "Rater1", "Rater2")
    assert isinstance(result, PairwiseResult)
    assert result.other_rater == "Rater2"
    assert 0 <= result.kripp_alpha <= 1
    assert -1 <= result.cohen_kappa <= 1
    assert 0 <= result.percent_agreement <= 100
    assert 0 <= result.ones_agreement <= 100
    assert 0 <= result.fives_agreement <= 100
    assert 0 <= result.mad <= 4
    assert -1 <= result.correlation <= 1


def test_calculate_pairwise_metrics_for_all(sample_data):
    results = calculate_pairwise_metrics_for_all(sample_data)
    assert isinstance(results, dict)
    assert len(results) == 3  # 3 raters
    for rater, pairwise_results in results.items():
        assert len(pairwise_results) == 2  # Each rater compared to 2 others
        assert all(isinstance(r, PairwiseResult) for r in pairwise_results)


def test_calculate_overall_metrics(sample_data):
    metrics = calculate_overall_metrics(sample_data)
    assert isinstance(metrics, list)
    assert len(metrics) == 6  # Header + 5 metrics
    assert all(len(row) == 3 for row in metrics)  # Each row has 3 columns


@pytest.mark.parametrize(
    "input_data, expected_agreement",
    [
        (pd.DataFrame({"R1": [1, 2, 3], "R2": [1, 2, 3]}), 100.0),
        (pd.DataFrame({"R1": [1, 2, 3], "R2": [1, 2, 4]}), 66.67),
        (pd.DataFrame({"R1": [1, 2, 3], "R2": [3, 2, 1]}), 33.33),
    ],
)
def test_calculate_overall_percent_agreement_parametrized(
    input_data, expected_agreement
):
    agreement = calculate_overall_percent_agreement(input_data)
    assert pytest.approx(agreement, 0.01) == expected_agreement


def test_empty_dataframe():
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError):
        calculate_krippendorff_alpha(empty_df)
    with pytest.raises(ValueError):
        calculate_fleiss_kappa(empty_df)
    with pytest.raises(ValueError):
        calculate_kendalls_w(empty_df)
    with pytest.raises(ValueError):
        calculate_mean_absolute_difference(empty_df)
    with pytest.raises(ValueError):
        calculate_overall_percent_agreement(empty_df)


def test_single_rater():
    single_rater_df = pd.DataFrame({"R1": [1, 2, 3, 4, 5]})
    with pytest.raises(ValueError):
        calculate_krippendorff_alpha(single_rater_df)
    with pytest.raises(ValueError):
        calculate_fleiss_kappa(single_rater_df)
    assert calculate_overall_percent_agreement(single_rater_df) == 100.0


def test_all_same_ratings():
    same_ratings_df = pd.DataFrame(
        {
            "R1": [3, 3, 3, 3, 3],
            "R2": [3, 3, 3, 3, 3],
            "R3": [3, 3, 3, 3, 3],
        }
    )
    assert calculate_krippendorff_alpha(same_ratings_df) == 1.0
    assert calculate_fleiss_kappa(same_ratings_df) == 1.0
    assert calculate_overall_percent_agreement(same_ratings_df) == 100.0
    assert calculate_kendalls_w(same_ratings_df) == 1.0
    assert calculate_mean_absolute_difference(same_ratings_df) == 0.0


def test_all_different_ratings():
    diff_ratings_df = pd.DataFrame(
        {
            "R1": [1, 2, 3, 4, 5],
            "R2": [2, 3, 4, 5, 1],
            "R3": [3, 4, 5, 1, 2],
        }
    )
    assert calculate_krippendorff_alpha(diff_ratings_df) < 0
    assert calculate_fleiss_kappa(diff_ratings_df) < 0
    assert calculate_overall_percent_agreement(diff_ratings_df) == 0.0
    assert (
        calculate_kendalls_w(diff_ratings_df) > 0
        and calculate_kendalls_w(diff_ratings_df) < 1
    )
    assert calculate_mean_absolute_difference(diff_ratings_df) > 0


@patch("rater_metric_calculator.calculate_rater_metrics.analyze_with_llm")
@patch("rater_metric_calculator.calculate_rater_metrics.get_markdown_report")
@patch("rater_metric_calculator.calculate_rater_metrics.pd.read_csv")
@patch("builtins.open", new_callable=mock_open)
def test_main_with_llm(
    mock_file_open, mock_read_csv, mock_get_markdown_report, mock_analyze_with_llm
):
    mock_read_csv.return_value = pd.DataFrame(
        {"Rater1": [1, 2, 3], "Rater2": [1, 2, 3]}
    )
    mock_get_markdown_report.return_value.get_md_text.return_value = "Markdown content"
    mock_analyze_with_llm.return_value = "LLM analysis result"

    main("input.csv", "output.md", use_llm=True, llm_report_file="llm_report.md")

    mock_read_csv.assert_called_once_with("input.csv")
    mock_get_markdown_report.assert_called_once()
    mock_analyze_with_llm.assert_called_once_with("Markdown content", None)

    mock_file_open.assert_called_with("llm_report.md", "w")
    mock_file_open().write.assert_called_once_with("LLM analysis result")


@patch("rater_metric_calculator.calculate_rater_metrics.get_markdown_report")
@patch("rater_metric_calculator.calculate_rater_metrics.pd.read_csv")
def test_main_without_llm(mock_read_csv, mock_get_markdown_report):
    mock_read_csv.return_value = pd.DataFrame(
        {"Rater1": [1, 2, 3], "Rater2": [1, 2, 3]}
    )

    main("input.csv", "output.md")

    mock_read_csv.assert_called_once_with("input.csv")
    mock_get_markdown_report.assert_called_once()
    mock_get_markdown_report.return_value.create_md_file.assert_called_once()

    mock_get_markdown_report.return_value.get_md_text.assert_not_called()
