# rater-metric-calculator
[![CI](https://github.com/jrrobison1/rater-metric-calculator/actions/workflows/ci.yml/badge.svg)](https://github.com/jrrobison1/rater-metric-calculator/actions/workflows/ci.yml) 

rater-metric-calculator is a Python tool designed to calculate several inter-rater reliability metrics for multi-rater datasets. It provides a comprehensive set of metrics to evaluate agreement among raters, and prints the results to a convenient well-formatted Markdown report. This tool is ideal for researchers working with subjective ratings or annotations.

## Features

- Calculate various inter-rater reliability metrics:
  - Krippendorff's Alpha
  - Fleiss' Kappa
  - Kendall's W (Coefficient of Concordance)
  - Overall Percent Agreement
  - Mean Absolute Difference
- Generate pairwise comparisons between raters
- Calculate rating distributions
- Output results in a well-formatted Markdown report

## Installation

To install the Rater Metric Calculator, you'll need Python 3.10 or higher. We recommend using [Poetry](https://python-poetry.org/) for dependency management.

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/rater-metric-calculator.git
   cd rater-metric-calculator
   ```

2. Install dependencies using Poetry:
   ```
   poetry install
   ```

## Usage

To use the Rater Metric Calculator, run the following command:

```
poetry run python rater_metric_calculator/calculate_rater_metrics.py <input_csv_file> <output_md_file>
```

Replace `<input_csv_file>` with the path to your CSV file containing rating data, and `<output_md_file>` with the desired path for the output Markdown report.

### Input Format

The input CSV file should have the following format:
- Each column represents a rater
- Each row represents an item being rated
- Ratings should be integers between 1 and 5 (inclusive)

Example:
```
Rater1,Rater2,Rater3
1,2,1
3,3,2
5,4,5
2,2,3
```

### Output

The script generates a Markdown file containing:
1. Overall metrics (Krippendorff's Alpha, Fleiss' Kappa, etc.)
2. Rating distribution for each rater
3. Pairwise comparisons between raters

## Running Tests

To run the test suite, use the following command:

```
poetry run pytest
```

## Contributing

Contributions to the Rater Metric Calculator are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project uses several scientific Python libraries, including pandas, numpy, scipy, and statsmodels.
- Special thanks to the developers of the krippendorff and mdutils packages.
