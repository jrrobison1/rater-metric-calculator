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
- Generate an AI-powered interpretation report using Claude 3.5 Sonnet (optional)

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
poetry run python rater_metric_calculator/calculate_rater_metrics.py <input_csv_file> <output_md_file> [--use-llm] [--llm-report-file <llm_report_file>] [--prompt-file <prompt_file>]
```

Replace `<input_csv_file>` with the path to your CSV file containing rating data, and `<output_md_file>` with the desired path for the output Markdown report.

To generate an AI-powered interpretation report using Claude 3.5 Sonnet, add the `--use-llm` flag to the command. When using this flag, you must also specify the `--llm-report-file` option with the desired path for the LLM report output file.

### LLM Report Generation

When the `--use-llm` flag is used, the script will:
1. Generate the standard metrics report
2. Use the Claude 3.5 Sonnet model to analyze the report
3. Save the AI-generated interpretation to the file specified by `--llm-report-file`

Example command with LLM report generation:
```
poetry run python rater_metric_calculator/calculate_rater_metrics.py input_data.csv output_report.md --use-llm --llm-report-file llm_interpretation.md
```

You can also specify a custom prompt file for the LLM analysis using the `--prompt-file` option:
```
poetry run python rater_metric_calculator/calculate_rater_metrics.py input_data.csv output_report.md --use-llm --llm-report-file llm_interpretation.md --prompt-file custom_prompt.txt
```

If no prompt file is specified, the script will use a default prompt for the analysis.

Note: To use the LLM report generation feature, you need to set the `ANTHROPIC_API_KEY` environment variable with your Anthropic API key.

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

You can optionally generate an AI-powered interpretation report using Claude 3.5 Sonnet, which provides insights and analysis based on the calculated metrics.

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