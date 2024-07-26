from dataclasses import dataclass
from sys import argv
from typing import Dict, List

import krippendorff
import numpy as np
import pandas as pd
from mdutils.mdutils import MdUtils
from scipy import stats
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa

MIN_RATING = 1
MAX_RATING = 5
RATING_RANGE = range(MIN_RATING, MAX_RATING + 1)
PAIRWISE_TABLE_HEADER = [
    "Other Rater",
    "Krippendorff Alpha",
    "Cohen Kappa",
    "% Exact Agreements",
    "% Agreement on 1's",
    "% Agreement on 5's",
    "Mean Absolute Score Difference",
    "Spearman Correlation",
]


def calculate_krippendorff_alpha(data: pd.DataFrame) -> float:
    """
    Calculate Krippendorff's alpha for inter-rater reliability.

    Args:
        data (pd.DataFrame): DataFrame containing ratings from multiple raters.

    Returns:
        float: Krippendorff's alpha value. Returns 1.0 if all ratings are equal.

    Raises:
        ValueError: If the input DataFrame is empty.
    """
    if data.empty:
        raise ValueError("Input DataFrame is empty")
    if data.nunique().eq(1).all():
        return 1.0

    return krippendorff.alpha(
        reliability_data=data.T.values, level_of_measurement="ordinal"
    )


def calculate_fleiss_kappa(data: pd.DataFrame) -> float:
    """
    Calculate Fleiss' kappa for inter-rater agreement.

    Args:
        data (pd.DataFrame): DataFrame containing ratings from multiple raters.

    Returns:
        float: Fleiss' kappa value. Returns 1.0 if all ratings are equal.

    Raises:
        ValueError: If there's one rater or no raters in the data.
    """
    if data.shape[1] < 2:
        raise ValueError("Fleiss' kappa requires at least two raters.")
    if data.nunique().eq(1).all():
        return 1.0

    def fleiss_kappa_input(data: pd.DataFrame) -> np.ndarray:
        ratings = []
        for i in range(data.shape[0]):
            row = data.iloc[i].value_counts().reindex(RATING_RANGE, fill_value=0).values
            ratings.append(row)
        return ratings

    ratings = fleiss_kappa_input(data)

    return fleiss_kappa(ratings)


def calculate_overall_percent_agreement(data: pd.DataFrame) -> float:
    """
    Calculate the overall percentage of exact agreements between raters.

    Args:
        data (pd.DataFrame): DataFrame containing ratings from multiple raters.

    Returns:
        float: Percentage of exact agreements.

    Raises:
        ValueError: If the input DataFrame is empty.
    """
    if data.empty:
        raise ValueError("Input DataFrame is empty")
    total_agreements = sum(data.nunique(axis=1) == 1)

    return (total_agreements / len(data)) * 100


def calculate_kendalls_w(data: pd.DataFrame) -> float:
    """
    Calculate Kendall's W (coefficient of concordance) for inter-rater agreement.

    Args:
        data (pd.DataFrame): DataFrame containing ratings from multiple raters.

    Returns:
        float: Kendall's W value. Returns 1.0 if all ratings are equal.

    Raises:
        ValueError: If the input DataFrame is empty.
    """
    if data.empty:
        raise ValueError("Input DataFrame is empty")
    if data.nunique().eq(1).all():
        return 1.0
    n = data.shape[0]  # number of subjects
    k = data.shape[1]  # number of raters

    # Calculate the sum of ranks for each subject
    rank_sums = data.rank().sum(axis=1)

    # Calculate the sum of squared deviations
    s = np.sum((rank_sums - np.mean(rank_sums)) ** 2)

    # Calculate Kendall's W
    w = (12 * s) / (k**2 * (n**3 - n))

    return w


def calculate_mean_absolute_difference(data: pd.DataFrame) -> float:
    """
    Calculate the mean absolute difference between all pairs of raters.

    Args:
        data (pd.DataFrame): DataFrame containing ratings from multiple raters.

    Returns:
        float: Mean absolute difference.

    Raises:
        ValueError: If the input DataFrame is empty.
    """
    if data.empty:
        raise ValueError("Input DataFrame is empty")

    return np.mean(
        [
            np.abs(data[col1] - data[col2]).mean()
            for i, col1 in enumerate(data.columns)
            for col2 in data.columns[i + 1 :]
        ]
    )


def calculate_rating_distribution(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the distribution of ratings for each rater.

    Args:
        data (pd.DataFrame): DataFrame containing ratings from multiple raters.

    Returns:
        pd.DataFrame: DataFrame containing the distribution of ratings for each rater.
    """

    # Create a function to calculate distribution and fill missing values with 0
    def distribution_with_zeros(x):
        dist = x.value_counts(normalize=True).sort_index() * 100
        return dist.reindex(RATING_RANGE, fill_value=0)

    return data.apply(distribution_with_zeros)


@dataclass
class PairwiseResult:
    """
    Dataclass to store pairwise comparison results between two raters.
    """

    other_rater: str
    kripp_alpha: float
    cohen_kappa: float
    percent_agreement: float
    ones_agreement: float
    fives_agreement: float
    mad: float
    correlation: float


def calculate_pairwise_metrics(
    data: pd.DataFrame, rater: str, other_rater: str
) -> PairwiseResult:
    """
    Calculate pairwise metrics between two raters.

    Args:
        data (pd.DataFrame): DataFrame containing ratings from multiple raters.
        rater (str): Name of the first rater.
        other_rater (str): Name of the second rater.

    Returns:
        PairwiseResult: Object containing pairwise comparison metrics.
    """
    pairwise_kripp_alpha = krippendorff.alpha(
        reliability_data=data[[rater, other_rater]].T.values,
        level_of_measurement="ordinal",
    )
    pairwise_cohen_kappa = cohen_kappa_score(
        data[rater], data[other_rater], weights="linear"
    )
    pairwise_ones_agreement = (
        np.mean((data[rater] == MIN_RATING) & (data[other_rater] == MIN_RATING)) * 100
    )
    pairwise_fives_agreement = (
        np.mean((data[rater] == MAX_RATING) & (data[other_rater] == MAX_RATING)) * 100
    )
    percent_agreement = np.mean(data[rater] == data[other_rater]) * 100
    mad = np.mean(np.abs(data[rater] - data[other_rater]))
    correlation = stats.spearmanr(data[rater], data[other_rater])[0]

    return PairwiseResult(
        other_rater,
        pairwise_kripp_alpha,
        pairwise_cohen_kappa,
        percent_agreement,
        pairwise_ones_agreement,
        pairwise_fives_agreement,
        mad,
        correlation,
    )


def calculate_pairwise_metrics_for_all(
    data: pd.DataFrame,
) -> Dict[str, List[PairwiseResult]]:
    """
    Calculate pairwise metrics for all raters in the dataset.

    Args:
        data (pd.DataFrame): DataFrame containing ratings from multiple raters.

    Returns:
        Dict[str, List[PairwiseResult]]: Dictionary with raters as keys and lists of PairwiseResult objects as values.
    """
    raters = data.columns
    results = {}

    for rater in raters:
        pairwise_results = []
        for other_rater in raters:
            if rater != other_rater:
                metrics = calculate_pairwise_metrics(data, rater, other_rater)
                pairwise_results.append(metrics)
        pairwise_results.sort(key=lambda x: x.kripp_alpha, reverse=True)
        results[rater] = pairwise_results

    return results


def calculate_overall_metrics(data: pd.DataFrame) -> List[List[str]]:
    """
    Calculate overall metrics for the entire dataset.

    Args:
        data (pd.DataFrame): DataFrame containing ratings from multiple raters.

    Returns:
        List[List[str]]: List of lists containing overall metric names, values, and notes.
    """
    kripp_alpha = calculate_krippendorff_alpha(data)
    fleiss_kappa_val = calculate_fleiss_kappa(data)
    overall_percent_agreement = calculate_overall_percent_agreement(data)
    kendall_w = calculate_kendalls_w(data)

    # Overall metrics
    overall_metrics = [
        ["Metric", "Value", "Notes"],
        ["Krippendorff's Alpha", f"{kripp_alpha:.3f}", ""],
        [
            "Fleiss' Kappa",
            f"{fleiss_kappa_val:.3f}",
            "Warning: treating ordinal data as nominal",
        ],
        ["Kendall's W", f"{kendall_w:.3f}", ""],
        ["Percentage of Exact Agreements", f"{overall_percent_agreement:.1f}%", ""],
        [
            "Mean Absolute Score Difference",
            f"{calculate_mean_absolute_difference(data):.3f}",
            f"Average absolute difference in scores. Range: 0-{MAX_RATING - MIN_RATING}. Lower is better",
        ],
    ]

    return overall_metrics


def write_markdown(
    overall_metrics: List[List[str]],
    ratings_distribution: pd.DataFrame,
    pairwise_metrics: Dict[str, List[PairwiseResult]],
    output_file: str,
) -> None:
    """
    Write the calculated metrics to a markdown file.

    Args:
        overall_metrics (List[List[str]]): Overall metrics data.
        ratings_distribution (pd.DataFrame): Distribution of ratings for each rater.
        pairwise_metrics (Dict[str, List[PairwiseResult]]): Pairwise metrics for all raters.
        output_file (str): Path to the output markdown file.
    """
    md_file = MdUtils(file_name=output_file, title="Rater Metrics Report")

    md_file.new_header(level=2, title="Overall Metrics", add_table_of_contents="n")
    md_file.new_table(
        columns=len(overall_metrics[0]),
        rows=len(overall_metrics),
        text=sum(overall_metrics, []),
        text_align="left",
    )

    md_file.new_header(
        level=2, title="Ratings Distribution (%)", add_table_of_contents="n"
    )
    md_file.new_table(
        columns=len(ratings_distribution.columns) + 1,
        rows=len(ratings_distribution.index) + 1,
        text=["Rating"]
        + list(ratings_distribution.columns)
        + sum(
            [
                [str(idx)] + [f"{val:.1f}%" for val in row]
                for idx, row in ratings_distribution.iterrows()
            ],
            [],
        ),
    )

    md_file.new_header(level=2, title="Pairwise", add_table_of_contents="n")
    for rater, pairwise_results in pairwise_metrics.items():
        md_file.new_header(level=3, title=f"Rater: {rater}", add_table_of_contents="n")
        table_data = [] + PAIRWISE_TABLE_HEADER
        for result in pairwise_results:
            table_data.extend(
                [
                    result.other_rater,
                    f"{result.kripp_alpha:.3f}",
                    f"{result.cohen_kappa:.3f}",
                    f"{result.percent_agreement:.1f}%",
                    f"{result.ones_agreement:.1f}%",
                    f"{result.fives_agreement:.1f}%",
                    f"{result.mad:.1f}",
                    f"{result.correlation:.1f}",
                ]
            )
        md_file.new_table(columns=8, rows=len(pairwise_results) + 1, text=table_data)

    md_file.create_md_file()


def main(input_file: str, output_file: str) -> None:
    """
    Main function to calculate and display inter-rater reliability metrics.

    Args:
        input_file (str): Path to the CSV file containing rating data.
        output_file (str): Path to the output markdown file.
    """
    data = pd.read_csv(input_file)

    overall_metrics = calculate_overall_metrics(data)
    ratings_distribution = calculate_rating_distribution(data)
    pairwise_metrics = calculate_pairwise_metrics_for_all(data)

    write_markdown(overall_metrics, ratings_distribution, pairwise_metrics, output_file)


if __name__ == "__main__":
    if len(argv) != 3:
        print(
            "Usage: python calculate_rater_metrics.py <input_csv_file> <output_md_file>"
        )
        exit(1)
    main(argv[1], argv[2])
