# Iowa Liquor Sales Data Synthesizer

This Python tool generates synthetic retail sales data based on statistical characteristics of real data.

## Features

- Analyzes original CSV data for statistical patterns:
  - Distribution types and parameters
  - Outlier detection
  - Day-of-week and seasonal patterns
  - Correlations between numeric variables
  - Value distributions for categorical variables
- Generates synthetic data that mirrors these characteristics
- Handles large files through chunking and sampling
- Preserves correlations between variables
- Maintains day-of-week sales patterns
- Applies realistic seasonality effects
- Creates detailed visualizations comparing original and synthetic data
- Produces a comprehensive summary report

## Requirements

- Python 3.6+
- Required packages:
  - pandas
  - numpy
  - matplotlib
  - scipy

## Usage

1. Place your CSV file in the same directory as the script (default expects "Iowa Liquor Sales 2024.csv")

2. Run the script with any of these command-line options:

```bash
# Default: Generate new synthetic data
python analyze_and_generate.py

# Analyze existing synthetic data file instead of generating new data
python analyze_and_generate.py --analyze-existing

# Only compare original and synthetic data (if synthetic data already exists)
python analyze_and_generate.py --compare-only  

# Use detailed (slower but more accurate) distribution detection
python analyze_and_generate.py --fast-distribution False
```

3. For custom parameters (input/output files, date ranges), use the example script:

```bash
python example.py --input "your_data.csv" --output "synthetic_output.csv" --start-date "2025-07-01" --end-date "2025-07-07"
```

4. The script will:
   - Analyze the original data
   - Generate synthetic data for the specified date range (default: June 1-6, 2025)
   - Save the synthetic data to the output file (default: "Iowa_Liquor_Sales_Synthetic_Jun2025.csv")
   - Create visualization plots in the "figures" directory
   - Generate a summary report "synthetic_data_report.txt"
   - Provide a detailed comparison between original and synthetic data

## Output Files

- **Iowa_Liquor_Sales_Synthetic_Jun2025.csv**: The synthetic data
- **synthetic_data_report.txt**: Summary report comparing original and synthetic data
- **figures/**: Directory containing visualization plots:
  - Distribution comparisons
  - Q-Q plots
  - Boxplots showing outlier distributions
  - Day-of-week pattern comparisons

## Advanced Features

- **High-Performance Data Generation**: 
  - Uses batch processing for optimal CPU utilization
  - Pre-computes distributions and statistical properties to avoid redundant calculations
  - Optimizes memory usage with garbage collection for large datasets
- **Detailed Progress Reporting**: Shows real-time progress updates during data generation with records/second metrics
- **Performance Timing**: Measures and reports processing time for each operation
- **Fast Distribution Detection**: Uses statistical moments (skewness, kurtosis) to quickly determine likely distributions
- **Correlation Preservation**: Maintains relationships between numeric variables using Cholesky decomposition
- **Outlier Generation**: Controls the frequency and magnitude of outliers to match the original data
- **Seasonality Modeling**: Adjusts values based on day-of-week patterns (weekends vs. weekdays)
- **Comparative Analysis**: Directly compare synthetic data with original data to validate quality
- **Large File Handling**: Processes large files in chunks to manage memory usage
- **Performance Metrics**: Reports processing time for each operation to identify bottlenecks

## Script Descriptions

- **analyze_and_generate.py**: Main script with all functionality
- **example.py**: Command-line interface for custom parameters
- **test_installation.py**: Verifies all dependencies are correctly installed
- **requirements.txt**: Lists required Python packages

## Customization

You can modify the scripts to:
- Change the date range for synthetic data
- Adjust seasonality factors
- Modify outlier generation rates
- Change the input/output file names
- Customize the analysis and comparison metrics
