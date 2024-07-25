from sys import argv
import numpy as np
import pandas as pd
import krippendorff
from scipy import stats
from statsmodels.stats.inter_rater import fleiss_kappa
from sklearn.metrics import cohen_kappa_score
from dataclasses import dataclass
from tabulate import tabulate

# Constants
MIN_RATING = 1
MAX_RATING = 5
RATING_RANGE = range(MIN_RATING, MAX_RATING + 1)


def calculate_krippendorff_alpha(data: pd.DataFrame) -> float:
    """
    Calculate Krippendorff's alpha for inter-rater reliability.

    Args:
        data (pd.DataFrame): DataFrame containing ratings from multiple raters.

    Returns:
        float: Krippendorff's alpha value.
    """
    return krippendorff.alpha(
        reliability_data=data.T.values, level_of_measurement="ordinal"
    )


def calculate_fleiss_kappa(data: pd.DataFrame) -> float:
    """
    Calculate Fleiss' kappa for inter-rater agreement.

    Args:
        data (pd.DataFrame): DataFrame containing ratings from multiple raters.

    Returns:
        float: Fleiss' kappa value.
    """

    def fleiss_kappa_input(data: pd.DataFrame) -> np.ndarray:
        """
        Prepare input data for Fleiss' kappa calculation.

        Args:
            data (pd.DataFrame): DataFrame containing ratings from multiple raters.

        Returns:
            np.ndarray: Array of rating counts for each item and category.
        """
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
    """
    total_agreements = sum(data.nunique(axis=1) == 1)
    return (total_agreements / len(data)) * 100


def calculate_kendalls_w(data: pd.DataFrame) -> float:
    """
    Calculate Kendall's W (coefficient of concordance) for inter-rater agreement.

    Args:
        data (pd.DataFrame): DataFrame containing ratings from multiple raters.

    Returns:
        float: Kendall's W value.
    """
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
    """
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


# Overall calculations
def print_overall_metrics(data: pd.DataFrame) -> None:
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

    print("## Overall Metrics")
    print(tabulate(overall_metrics, headers="firstrow", tablefmt="pipe"))


def print_ratings_distribution(data: pd.DataFrame):
    print("\n## Ratings Distribution (%)")
    distribution = calculate_rating_distribution(data)
    print(distribution.to_markdown(floatfmt=".2f"))


# Pairwise calculations for each rater
def print_pairwise_metrics(data: pd.DataFrame) -> None:
    raters = data.columns
    results = []

    for rater in raters:
        pairwise_results = []
        for other_rater in raters:
            if rater != other_rater:
                metrics = calculate_pairwise_metrics(data, rater, other_rater)
                pairwise_results.append(metrics)
        pairwise_results.sort(key=lambda x: x.kripp_alpha, reverse=True)
        results.append((rater, pairwise_results))

    print("\n## Pairwise Metrics")

    # Display results using tabulate
    for rater, pairwise_results in results:
        print(f"### Rater: {rater}")

        table_data = [
            [
                "Other Rater",
                "Krippendorff Alpha",
                "Cohen Kappa",
                "% Exact Agreements",
                "% Agreement on 1's",
                "% Agreement on 5's",
                "Mean Absolute Score Difference",
                "Spearman Correlation",
            ]
        ]

        for result in pairwise_results:
            table_data.append(
                [
                    result.other_rater,
                    f"{result.kripp_alpha:.3f}",
                    f"{result.cohen_kappa:.3f}",
                    f"{result.percent_agreement:.1f}",
                    f"{result.ones_agreement:.1f}",
                    f"{result.fives_agreement:.1f}",
                    f"{result.mad:.1f}",
                    f"{result.correlation:.1f}",
                ]
            )

        print(tabulate(table_data, headers="firstrow", tablefmt="pipe"))
        print("\n")


def main(file_name: str) -> None:
    """
    Main function to calculate and display inter-rater reliability metrics.

    Args:
        file_name (str): Path to the CSV file containing rating data.
    """
    # Load the data
    data = pd.read_csv(file_name)

    print_overall_metrics(data)
    print_ratings_distribution(data)
    print_pairwise_metrics(data)


if __name__ == "__main__":
    main(argv[1])
