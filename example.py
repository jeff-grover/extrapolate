#!/usr/bin/env python3
"""
Iowa Liquor Sales Data Synthesizer - Example Script

This script demonstrates how to use the analyze_and_generate.py
tool with custom parameters. It allows you to specify different
date ranges and input/output files.
"""

import os
import sys
import argparse
from datetime import datetime

# Try to import from the main script
try:
    from analyze_and_generate import analyze_csv, analyze_time_patterns, generate_synthetic_data
except ImportError:
    print("Error: analyze_and_generate.py must be in the same directory")
    sys.exit(1)

# Ensure scipy is available for distribution detection
try:
    import scipy.stats
except ImportError:
    print("Warning: scipy not found. Install with 'pip install scipy' for distribution analysis")

def main():
    """Parse arguments and run the data synthesis"""
    
    parser = argparse.ArgumentParser(description='Generate synthetic Iowa liquor sales data')
    
    parser.add_argument('--input', '-i', default='Iowa Liquor Sales 2024.csv',
                        help='Input CSV file (default: Iowa Liquor Sales 2024.csv)')
    
    parser.add_argument('--output', '-o', default='Iowa_Liquor_Sales_Synthetic_Output.csv',
                        help='Output CSV file (default: Iowa_Liquor_Sales_Synthetic_Output.csv)')
    
    parser.add_argument('--start-date', '-s', default='2025-06-01',
                        help='Start date for synthetic data (YYYY-MM-DD) (default: 2025-06-01)')
    
    parser.add_argument('--end-date', '-e', default='2025-06-06',
                        help='End date for synthetic data (YYYY-MM-DD) (default: 2025-06-06)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found")
        sys.exit(1)
    
    # Run analysis
    print(f"Analyzing input file: {args.input}")
    analysis = analyze_csv(args.input)
    
    # Analyze time patterns if date column exists
    if analysis['date_columns'] and len(analysis['date_columns']) > 0:
        date_col = analysis['date_columns'][0]
        time_patterns = analyze_time_patterns(analysis['dataframe'], date_col)
        if time_patterns:
            analysis['time_patterns'] = time_patterns
    
    # Calculate outliers
    outliers = {}
    for col in analysis['numeric_columns']:
        Q1 = analysis['dataframe'][col].quantile(0.25)
        Q3 = analysis['dataframe'][col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_count = ((analysis['dataframe'][col] < lower_bound) | 
                        (analysis['dataframe'][col] > upper_bound)).sum()
        outlier_percent = (outlier_count / len(analysis['dataframe'])) * 100
        
        outliers[col] = {
            'count': outlier_count,
            'percent': outlier_percent,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
    
    analysis['outliers'] = outliers
    
    # Generate synthetic data
    print(f"Generating synthetic data from {args.start_date} to {args.end_date}")
    synthetic_df = generate_synthetic_data(
        analysis, 
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    # Save synthetic data
    print(f"Saving synthetic data to {args.output}")
    synthetic_df.to_csv(args.output, index=False)
    
    # Print summary
    print("\nSummary:")
    print(f"Original records: {len(analysis['dataframe'])}")
    print(f"Synthetic records: {len(synthetic_df)}")
    print(f"Date range: {args.start_date} to {args.end_date}")
    print(f"Output file: {args.output}")
    
    print("\nDone! To generate visualizations and a full report, run the main script:")
    print("python analyze_and_generate.py")

if __name__ == "__main__":
    main()
