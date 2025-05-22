import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import multiprocessing
from functools import partial
import time

def analyze_csv(csv_path):
    """Analyze the CSV file and extract statistical properties."""
    print(f"Loading data from: {csv_path}")
    
    # Get file size for progress tracking
    file_size = os.path.getsize(csv_path) / (1024 * 1024)  # Size in MB
    print(f"File size: {file_size:.2f} MB")
    
    # For large files, use chunking to analyze
    if file_size > 100:  # If file is greater than 100MB
        print("Large file detected, using chunked processing...")
        
        # First, read a small sample to get column types
        sample_df = pd.read_csv(csv_path, nrows=1000)
        column_types = {col: str if sample_df[col].dtype == 'object' else sample_df[col].dtype 
                        for col in sample_df.columns}
        
        # Initialize empty DataFrame with same structure
        df = pd.DataFrame(columns=sample_df.columns)
        
        # Read in chunks
        chunk_size = 100000
        chunks = pd.read_csv(csv_path, chunksize=chunk_size, dtype=column_types)
        
        total_rows = 0
        for i, chunk in enumerate(chunks):
            total_rows += len(chunk)
            print(f"Processing chunk {i+1}, total rows processed: {total_rows}")
            df = pd.concat([df, chunk])
            
            # For very large files, sample instead of processing everything
            if total_rows > 1000000:
                print("Very large file detected, using sampling...")
                break
    else:
        # For smaller files, load directly
        df = pd.read_csv(csv_path)
    
    print(f"Loaded data with {len(df)} rows and {len(df.columns)} columns")
    
    # Display basic info about the dataset
    print("\nBasic Information:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Try to identify date column
    date_cols = [col for col in df.columns if 'date' in col.lower()]
    if date_cols:
        print(f"\nPotential date column(s): {date_cols}")
        
        # Convert date column(s) to datetime
        for col in date_cols:
            try:
                df[col] = pd.to_datetime(df[col])
                print(f"  Date range for {col}: {df[col].min()} to {df[col].max()}")
            except Exception as e:
                print(f"  Error converting {col} to datetime: {e}")
    
    # Handle missing values
    print("\nChecking for missing values...")
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    missing_stats = pd.DataFrame({'Missing Values': missing_data, 
                                  'Percent Missing': missing_percent})
    print(missing_stats[missing_stats['Missing Values'] > 0])
    
    # Analyze numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"\nNumeric columns: {numeric_cols}")
    
    # Get statistics for numeric columns
    numeric_stats = df[numeric_cols].describe().T
    print("\nNumeric Column Statistics:")
    print(numeric_stats)
    
    # Detect outliers
    print("\nDetecting outliers in numeric columns...")
    outliers = {}
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        outlier_percent = (outlier_count / len(df)) * 100
        
        outliers[col] = {
            'count': outlier_count,
            'percent': outlier_percent,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
        
        print(f"  {col}: {outlier_count} outliers ({outlier_percent:.2f}%)")
    
    # Analyze categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    print(f"\nCategorical columns: {categorical_cols}")
    
    # Get value counts for categorical columns (limited)
    cat_data = {}
    for col in categorical_cols:
        value_counts = df[col].value_counts()
        cat_data[col] = {
            'unique_values': df[col].nunique(),
            'top_values': value_counts.head(5).to_dict()
        }
    
    print("\nCategorical Column Information:")
    for col, info in cat_data.items():
        print(f"  {col}: {info['unique_values']} unique values")
        print(f"    Top 5 values: {info['top_values']}")
    
    return {
        'dataframe': df,
        'date_columns': date_cols,
        'numeric_columns': numeric_cols,
        'numeric_stats': numeric_stats,
        'categorical_columns': categorical_cols,
        'categorical_data': cat_data
    }

def process_column_distribution(column_name, df, fast=True):
    """Process a single column's distribution detection (for parallel processing)."""
    start_time = time.time()
    data = df[column_name]
    dist_type = detect_distribution(data, fast=fast)
    elapsed = time.time() - start_time
    return column_name, dist_type, elapsed

def detect_distribution(data, fast=True):
    """Detect the most likely distribution for a data series."""
    # Remove NaNs
    data = data.dropna()
    
    # Skip if too few data points
    if len(data) < 10:
        return "insufficient data"
    
    if fast:
        # Fast method: use simple checks based on statistical moments
        # Sample the data if it's large (for speed)
        if len(data) > 1000:
            data = data.sample(1000)
        
        # Calculate moments
        mean = data.mean()
        median = data.median()
        skew = data.skew()
        kurtosis = data.kurtosis()
        min_val = data.min()
        
        # Simple heuristic checks
        if min_val < 0:
            # Data has negative values
            if abs(skew) < 0.5 and abs(kurtosis) < 1:
                return "normal"
            else:
                return "other"
        else:
            # Data is non-negative
            mean_median_ratio = mean / median if median != 0 else float('inf')
            
            if abs(mean_median_ratio - 1) < 0.1 and abs(skew) < 0.5:
                return "normal"
            elif skew > 1 and mean_median_ratio > 1.2:
                return "lognormal" if min_val == 0 else "gamma"
            elif skew < -0.5:
                return "beta"
            elif abs(skew) < 0.5 and mean_median_ratio < 0.9:
                return "uniform"
            else:
                return "unknown"
    else:
        # Detailed method: use statistical tests (slower)
        import scipy.stats as stats
        
        # Define distributions to test
        distributions = [
            ("normal", stats.norm),
            ("lognormal", stats.lognorm),
            ("exponential", stats.expon),
            ("gamma", stats.gamma),
            ("beta", stats.beta)
        ]
        
        # Sample the data if it's large
        if len(data) > 1000:
            data = data.sample(1000)
        
        # Test each distribution
        results = []
        for name, dist in distributions:
            try:
                # Fit distribution parameters
                params = dist.fit(data)
                # Calculate Kolmogorov-Smirnov test statistic
                ks_stat, p_value = stats.kstest(data, name, params)
                results.append((name, p_value))
            except Exception:
                continue
        
        # Sort by p-value (higher is better fit)
        results.sort(key=lambda x: x[1], reverse=True)
        
        if not results:
            return "unknown"
        
        # Return the best fit distribution
        return results[0][0]

def generate_synthetic_data(analysis, start_date='2025-06-01', end_date='2025-06-06'):
    """Generate synthetic data based on the analysis results."""
    print(f"\nGenerating synthetic data from {start_date} to {end_date}")
    
    # Convert to datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Extract dataframe and stats from analysis
    orig_df = analysis['dataframe']
    numeric_cols = analysis['numeric_columns']
    numeric_stats = analysis['numeric_stats']
    categorical_cols = analysis['categorical_columns']
    categorical_data = analysis['categorical_data']
    date_cols = analysis['date_columns']
    
    # Calculate correlation matrix for numeric columns
    print("Calculating correlations between numeric columns...")
    if len(numeric_cols) > 1:
        corr_matrix = orig_df[numeric_cols].corr()
        print("\nCorrelation matrix:")
        print(corr_matrix)
    else:
        corr_matrix = None
    
    # Determine number of records per day based on original data
    if date_cols and len(date_cols) > 0:
        # Use the first date column
        date_col = date_cols[0]
        try:
            # Count records per day in original data
            orig_df[date_col] = pd.to_datetime(orig_df[date_col])
            records_per_day = orig_df.groupby(orig_df[date_col].dt.date).size()
            avg_records_per_day = int(records_per_day.mean())
            std_records_per_day = int(records_per_day.std())
            
            print(f"Original data has an average of {avg_records_per_day} records per day (std: {std_records_per_day})")
            
            # Generate random number of records for each day
            records_count = {}
            for date in date_range:
                day_count = max(1, int(np.random.normal(avg_records_per_day, std_records_per_day)))
                records_count[date] = day_count
        except Exception as e:
            print(f"Error analyzing records per day: {e}")
            # Fallback to a reasonable number if date analysis fails
            records_count = {date: 100 for date in date_range}
    else:
        # Fallback if no date column is identified
        records_count = {date: 100 for date in date_range}
    
    # Create empty dataframe for synthetic data
    synthetic_data = []
    
    # Performance optimization: Pre-compute common values and move them outside loops
    print("Preparing for data generation (optimizing for speed)...")
    
    # Pre-compute categorical value distributions (this was a major bottleneck)
    categorical_distributions = {}
    for col in categorical_cols:
        value_counts = orig_df[col].value_counts(normalize=True)
        if len(value_counts) > 0:
            # Store both indices and probabilities as numpy arrays for faster access
            categorical_distributions[col] = {
                'indices': np.array(value_counts.index),
                'probs': np.array(value_counts.values)
            }
    
    # Pre-compute outlier information for each column
    outlier_info_dict = {}
    for col in numeric_cols:
        if 'outliers' in analysis and col in analysis['outliers']:
            outlier_info = analysis['outliers'][col]
            lower_bound = outlier_info['lower_bound']
            upper_bound = outlier_info['upper_bound']
            # Reduce the probability of generating outliers
            outlier_chance = outlier_info['percent'] / 100
            reduced_chance = outlier_chance * 0.8
        else:
            mean = numeric_stats.loc[col, 'mean']
            std = numeric_stats.loc[col, 'std']
            min_val = numeric_stats.loc[col, 'min']
            max_val = numeric_stats.loc[col, 'max']
            lower_bound = min_val
            upper_bound = max_val
            reduced_chance = 0.01
        
        outlier_info_dict[col] = {
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'reduced_chance': reduced_chance
        }
    
    # Pre-compute Cholesky decomposition if possible
    L = None
    if corr_matrix is not None and len(numeric_cols) > 1:
        try:
            L = np.linalg.cholesky(corr_matrix)
            print("✓ Pre-computed Cholesky decomposition for correlation matrix")
        except np.linalg.LinAlgError:
            print("⚠ Could not apply correlations (matrix not positive definite)")
    
# Define a global batch processing function (outside of any other function) to avoid pickling issues
def process_batch(batch_info):
    """
    Generate a batch of records - must be defined at module level for multiprocessing
    
    Parameters:
    -----------
    batch_info : tuple
        A tuple containing:
        - batch_num: The batch number
        - batch_size: Size of the batch
        - total_batches: Total number of batches
        - batch_start: Starting index
        - batch_end: Ending index
        - context: Dictionary with all needed data for generation
    """
    (batch_num, batch_size, total_batches, batch_start, batch_end, context) = batch_info
    
    # Extract context data
    date = context['date']
    date_cols = context['date_cols']
    numeric_cols = context['numeric_cols']
    categorical_distributions = context['categorical_distributions']
    numeric_stats = context['numeric_stats']
    L = context['L']
    day_factor = context['day_factor']
    month_factor = context['month_factor']
    outlier_info_dict = context['outlier_info_dict']
    categorical_cols = context['categorical_cols']
    
    # Local start time for this batch
    batch_start_time = time.time()
    current_batch_size = batch_end - batch_start
    
    # Create batch records structure
    batch_records = []
    
    # Generate all numeric data at once using vectorized operations
    # First generate uncorrelated normal values for all records in batch
    uncorrelated_batch = {col: np.random.normal(0, 1, current_batch_size) for col in numeric_cols}
    
    # Apply correlations if available
    if L is not None:
        for i in range(current_batch_size):
            # Extract values for this record
            uncorrelated_matrix = np.array([uncorrelated_batch[col][i] for col in numeric_cols])
            # Apply correlation
            correlated_matrix = np.dot(L, uncorrelated_matrix)
            # Update values
            for j, col in enumerate(numeric_cols):
                uncorrelated_batch[col][i] = correlated_matrix[j]
    
    # Generate all records in batch
    for i in range(current_batch_size):
        record = {}
        
        # Add date
        if date_cols and len(date_cols) > 0:
            record[date_cols[0]] = date
        
        # Add numeric data based on statistics
        for col in numeric_cols:
            mean = numeric_stats.loc[col, 'mean']
            std = numeric_stats.loc[col, 'std']
            min_val = numeric_stats.loc[col, 'min']
            max_val = numeric_stats.loc[col, 'max']
            
            # Get pre-computed outlier information
            outlier_info = outlier_info_dict[col]
            lower_bound = outlier_info['lower_bound']
            upper_bound = outlier_info['upper_bound']
            reduced_chance = outlier_info['reduced_chance']
            
            # Apply the value with seasonality
            value = mean + (uncorrelated_batch[col][i] * std * day_factor * month_factor)
            
            # Occasionally generate outliers
            if np.random.random() < reduced_chance:
                # Generate a value outside the normal range
                if np.random.random() < 0.5:
                    # Generate lower outlier
                    value = lower_bound - (np.random.random() * (lower_bound - min_val) * 0.5)
                else:
                    # Generate upper outlier
                    value = upper_bound + (np.random.random() * (max_val - upper_bound) * 0.5)
            
            # Clip to min/max
            value = np.clip(value, min_val, max_val)
            
            # Add to record
            record[col] = value
        
        # Add categorical data based on original distribution
        for col in categorical_cols:
            if col in categorical_distributions:
                # Use the pre-computed distributions (much faster)
                dist = categorical_distributions[col]
                record[col] = np.random.choice(dist['indices'], p=dist['probs'])
            else:
                # Handle empty value_counts
                record[col] = np.nan
        
        batch_records.append(record)
    
    batch_elapsed = time.time() - batch_start_time
    batch_records_per_sec = current_batch_size / batch_elapsed if batch_elapsed > 0 else 0
    
    return {
        'records': batch_records,
        'batch_num': batch_num,
        'size': current_batch_size,
        'elapsed': batch_elapsed,
        'records_per_sec': batch_records_per_sec
    }

def generate_synthetic_data(analysis, start_date='2025-06-01', end_date='2025-06-06'):
    """Generate synthetic data based on the analysis results."""
    print(f"\nGenerating synthetic data from {start_date} to {end_date}")
    
    # Convert to datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Extract dataframe and stats from analysis
    orig_df = analysis['dataframe']
    numeric_cols = analysis['numeric_columns']
    numeric_stats = analysis['numeric_stats']
    categorical_cols = analysis['categorical_columns']
    categorical_data = analysis['categorical_data']
    date_cols = analysis['date_columns']
    
    # Calculate correlation matrix for numeric columns
    print("Calculating correlations between numeric columns...")
    if len(numeric_cols) > 1:
        corr_matrix = orig_df[numeric_cols].corr()
        print("\nCorrelation matrix:")
        print(corr_matrix)
    else:
        corr_matrix = None
    
    # Determine number of records per day based on original data
    if date_cols and len(date_cols) > 0:
        # Use the first date column
        date_col = date_cols[0]
        try:
            # Count records per day in original data
            orig_df[date_col] = pd.to_datetime(orig_df[date_col])
            records_per_day = orig_df.groupby(orig_df[date_col].dt.date).size()
            avg_records_per_day = int(records_per_day.mean())
            std_records_per_day = int(records_per_day.std())
            
            print(f"Original data has an average of {avg_records_per_day} records per day (std: {std_records_per_day})")
            
            # Generate random number of records for each day
            records_count = {}
            for date in date_range:
                day_count = max(1, int(np.random.normal(avg_records_per_day, std_records_per_day)))
                records_count[date] = day_count
        except Exception as e:
            print(f"Error analyzing records per day: {e}")
            # Fallback to a reasonable number if date analysis fails
            records_count = {date: 100 for date in date_range}
    else:
        # Fallback if no date column is identified
        records_count = {date: 100 for date in date_range}
    
    # Create empty dataframe for synthetic data
    synthetic_data = []
    
    # Performance optimization: Pre-compute common values and move them outside loops
    print("Preparing for data generation (optimizing for speed)...")
    
    # Pre-compute categorical value distributions (this was a major bottleneck)
    categorical_distributions = {}
    for col in categorical_cols:
        value_counts = orig_df[col].value_counts(normalize=True)
        if len(value_counts) > 0:
            # Store both indices and probabilities as numpy arrays for faster access
            categorical_distributions[col] = {
                'indices': np.array(value_counts.index),
                'probs': np.array(value_counts.values)
            }
    
    # Pre-compute outlier information for each column
    outlier_info_dict = {}
    for col in numeric_cols:
        if 'outliers' in analysis and col in analysis['outliers']:
            outlier_info = analysis['outliers'][col]
            lower_bound = outlier_info['lower_bound']
            upper_bound = outlier_info['upper_bound']
            # Reduce the probability of generating outliers
            outlier_chance = outlier_info['percent'] / 100
            reduced_chance = outlier_chance * 0.8
        else:
            mean = numeric_stats.loc[col, 'mean']
            std = numeric_stats.loc[col, 'std']
            min_val = numeric_stats.loc[col, 'min']
            max_val = numeric_stats.loc[col, 'max']
            lower_bound = min_val
            upper_bound = max_val
            reduced_chance = 0.01
        
        outlier_info_dict[col] = {
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'reduced_chance': reduced_chance
        }
    
    # Pre-compute Cholesky decomposition if possible
    L = None
    if corr_matrix is not None and len(numeric_cols) > 1:
        try:
            L = np.linalg.cholesky(corr_matrix)
            print("✓ Pre-computed Cholesky decomposition for correlation matrix")
        except np.linalg.LinAlgError:
            print("⚠ Could not apply correlations (matrix not positive definite)")
    
    # We'll use numpy arrays for batch generation instead of record-by-record
    print("Generating data for each day with optimized approach...")
    
    # Generate all records
    start_time = time.time()
    
    # Process each day
    for date in date_range:
        day_start_time = time.time()
        num_records = records_count[date]
        print(f"Generating {num_records} records for {date.date()}")
        
        # Extract day-specific info
        day_of_week = date.dayofweek  # 0=Monday, 6=Sunday
        month = date.month
        
        # Apply seasonal factors based on day of week
        # Weekend effect: increase sales on Friday/Saturday, decrease on Sunday/Monday
        day_factor = 1.0
        if day_of_week == 4:  # Friday
            day_factor = 1.15
        elif day_of_week == 5:  # Saturday
            day_factor = 1.25
        elif day_of_week == 6:  # Sunday
            day_factor = 0.85
        elif day_of_week == 0:  # Monday
            day_factor = 0.90
        
        # Month seasonality (June is typically higher than average for liquor sales)
        month_factor = 1.1  # June boost for summer
        
        # Create a batch size that balances memory usage with performance
        # Use smaller batches to enable better parallelization
        # Aim for 10-20 batches to fully utilize CPU cores
        num_cpus = multiprocessing.cpu_count()
        target_batches = max(12, num_cpus * 2)  # Aim for at least 2 batches per CPU
        batch_size = max(1, num_records // target_batches)
        num_batches = (num_records + batch_size - 1) // batch_size  # Ceiling division
        
        # Progress tracking
        update_interval = max(1, min(100, num_records // 20))
        last_update_time = time.time()
        first_updates = [1, 5, 10, 25, 50, 100]
        first_updates = [x for x in first_updates if x < update_interval]
        
        print(f"  Starting generation for {date.date()} (using {num_cpus} CPU cores, processing in {num_batches} parallel batches)")
        
        # Create context dictionary with all data needed for batch processing
        # This avoids the need to pickle the enclosing function's variables
        batch_context = {
            'date': date,
            'date_cols': date_cols,
            'numeric_cols': numeric_cols,
            'categorical_cols': categorical_cols,
            'categorical_distributions': categorical_distributions,
            'numeric_stats': numeric_stats,
            'L': L,
            'day_factor': day_factor,
            'month_factor': month_factor,
            'outlier_info_dict': outlier_info_dict
        }
        
        # Prepare batch parameters for parallel processing
        batch_params = []
        for batch_num in range(num_batches):
            batch_start = batch_num * batch_size
            batch_end = min((batch_num + 1) * batch_size, num_records)
            # Include the context dictionary in each batch params
            batch_params.append((batch_num, batch_size, num_batches, batch_start, batch_end, batch_context))
        
        # Use multiprocessing pool to process batches in parallel
        day_records = []
        records_generated = 0
        
        with multiprocessing.Pool(processes=num_cpus) as pool:
            # Process batches in parallel
            results = pool.map(process_batch, batch_params)
            
            # Sort results by batch number to maintain order
            results.sort(key=lambda x: x['batch_num'])
            
            # Process results
            for result in results:
                batch_records = result['records']
                batch_size = result['size']
                batch_elapsed = result['elapsed']
                batch_records_per_sec = result['records_per_sec']
                
                # Extend day records with batch records
                day_records.extend(batch_records)
                
                # Update records generated count
                records_generated += batch_size
                
                # Print batch statistics
                progress = records_generated / num_records * 100
                print(f"  {date.date()}: Batch {result['batch_num']+1}/{num_batches} complete - {batch_size} records in {batch_elapsed:.2f}s ({batch_records_per_sec:.1f} records/sec)")
                
            # Print overall progress
            progress = 100.0  # All batches complete
            print(f"  {date.date()}: {records_generated}/{num_records} records ({progress:.1f}%)")
        
        # Add day records to synthetic data
        synthetic_data.extend(day_records)
            
        
        day_elapsed = time.time() - day_start_time
        day_records_per_sec = num_records / day_elapsed if day_elapsed > 0 else 0
        print(f"  Completed {date.date()}: {num_records} records in {day_elapsed:.2f}s ({day_records_per_sec:.1f} records/sec)")
    
    generation_time = time.time() - start_time
    total_records = sum(records_count.values())
    records_per_sec = total_records / generation_time if generation_time > 0 else 0
    print(f"Generated {total_records} records in {generation_time:.2f}s ({records_per_sec:.1f} records/sec)")
    
    # Convert to dataframe
    synthetic_df = pd.DataFrame(synthetic_data)
    
    # Match data types with original dataframe
    for col in synthetic_df.columns:
        if col in orig_df.columns:
            synthetic_df[col] = synthetic_df[col].astype(orig_df[col].dtype)
    
    print(f"Generated {len(synthetic_df)} synthetic records")
    return synthetic_df

def analyze_time_patterns(df, date_col):
    """Analyze time-based patterns in the original data."""
    print("\nAnalyzing time-based patterns...")
    
    try:
        # Ensure date column is datetime
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Add day of week
        df['day_of_week'] = df[date_col].dt.dayofweek
        
        # Analyze sales by day of week
        day_of_week_stats = df.groupby('day_of_week').agg({'size': 'count'})
        print("\nSales by day of week (0=Monday, 6=Sunday):")
        print(day_of_week_stats)
        
        # Analyze sales by month
        df['month'] = df[date_col].dt.month
        month_stats = df.groupby('month').agg({'size': 'count'})
        print("\nSales by month (1=January, 12=December):")
        print(month_stats)
        
        return {
            'day_of_week_stats': day_of_week_stats,
            'month_stats': month_stats
        }
    except Exception as e:
        print(f"Error analyzing time patterns: {e}")
        return None

def analyze_and_compare(original_df, synthetic_df, original_analysis=None):
    """Analyze and compare synthetic data with original data."""
    print("\nComparing synthetic data with original data:")
    
    # If no original analysis is provided, create one
    if original_analysis is None:
        # Extract basic stats from dataframes directly
        pass
    
    # Add day_of_week to synthetic data if needed for comparison
    date_cols = original_analysis.get('date_columns', []) if original_analysis else []
    if 'day_of_week' in original_df.columns and date_cols and len(date_cols) > 0:
        date_col = date_cols[0]
        if date_col in synthetic_df.columns and 'day_of_week' not in synthetic_df.columns:
            try:
                # Add day of week to synthetic data
                synthetic_df[date_col] = pd.to_datetime(synthetic_df[date_col])
                synthetic_df['day_of_week'] = synthetic_df[date_col].dt.dayofweek
                print("Added day_of_week column to synthetic data for comparison")
            except Exception as e:
                print(f"Could not add day_of_week column: {e}")
    
    # Compare basic statistics
    print("\nBasic Statistics Comparison:")
    numeric_cols = original_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter to only include columns that exist in both dataframes
    common_numeric_cols = [col for col in numeric_cols if col in synthetic_df.columns]
    if len(common_numeric_cols) < len(numeric_cols):
        print(f"Note: Only comparing {len(common_numeric_cols)} of {len(numeric_cols)} numeric columns that exist in both datasets.")
    
    for col in common_numeric_cols:
        print(f"\nColumn: {col}")
        orig_stats = original_df[col].describe()
        synth_stats = synthetic_df[col].describe()
        
        print(f"  Mean: {orig_stats['mean']:.2f} vs {synth_stats['mean']:.2f}")
        print(f"  Std : {orig_stats['std']:.2f} vs {synth_stats['std']:.2f}")
        print(f"  Min : {orig_stats['min']:.2f} vs {synth_stats['min']:.2f}")
        print(f"  25% : {orig_stats['25%']:.2f} vs {synth_stats['25%']:.2f}")
        print(f"  50% : {orig_stats['50%']:.2f} vs {synth_stats['50%']:.2f}")
        print(f"  75% : {orig_stats['75%']:.2f} vs {synth_stats['75%']:.2f}")
        print(f"  Max : {orig_stats['max']:.2f} vs {synth_stats['max']:.2f}")
        
        # Calculate percent difference
        if orig_stats['mean'] != 0:
            mean_diff = abs((synth_stats['mean'] - orig_stats['mean']) / orig_stats['mean'] * 100)
            print(f"  Mean difference: {mean_diff:.2f}%")
        
        if orig_stats['std'] != 0:
            std_diff = abs((synth_stats['std'] - orig_stats['std']) / orig_stats['std'] * 100)
            print(f"  Std difference: {std_diff:.2f}%")
    
    # Compare categorical distributions if any
    cat_cols = original_df.select_dtypes(include=['object']).columns.tolist()
    if cat_cols:
        print("\nCategorical Distribution Comparison:")
        for col in cat_cols:
            print(f"\nColumn: {col}")
            orig_counts = original_df[col].value_counts(normalize=True).head(5)
            synth_counts = synthetic_df[col].value_counts(normalize=True).head(5)
            
            print("  Original top categories:")
            for cat, freq in orig_counts.items():
                print(f"    {cat}: {freq:.4f}")
            
            print("  Synthetic top categories:")
            for cat, freq in synth_counts.items():
                print(f"    {cat}: {freq:.4f}")
    
    
    # Compare day of week distribution if applicable
    if 'day_of_week' in original_df.columns and 'day_of_week' in synthetic_df.columns:
        print("\nDay of Week Distribution:")
        orig_dow = original_df.groupby('day_of_week').size().reindex(range(7), fill_value=0)
        synth_dow = synthetic_df.groupby('day_of_week').size().reindex(range(7), fill_value=0)
        
        # Normalize
        orig_dow = orig_dow / orig_dow.sum()
        synth_dow = synth_dow / synth_dow.sum()
        
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for i, day in enumerate(days):
            print(f"  {day}: {orig_dow.get(i, 0):.4f} vs {synth_dow.get(i, 0):.4f}")
    
    # Return a report dictionary
    report = {
        'numeric_comparison': {col: {
            'original_mean': original_df[col].mean(),
            'synthetic_mean': synthetic_df[col].mean(),
            'original_std': original_df[col].std(),
            'synthetic_std': synthetic_df[col].std(),
        } for col in common_numeric_cols},
        'record_counts': {
            'original': len(original_df),
            'synthetic': len(synthetic_df)
        }
    }
    
    return report

def main():
    """Main function."""
    # Create figures directory if it doesn't exist
    if not os.path.exists('figures'):
        os.makedirs('figures')
    
    # Define input and output files
    input_file = "Iowa Liquor Sales 2024.csv"
    output_file = "Iowa_Liquor_Sales_Synthetic_Jun2025.csv"
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Generate and analyze synthetic sales data")
    parser.add_argument('--analyze-existing', action='store_true', 
                       help='Analyze existing synthetic data file instead of generating new data')
    parser.add_argument('--fast-distribution', action='store_true', default=True,
                       help='Use faster distribution detection (less rigorous)')
    parser.add_argument('--compare-only', action='store_true',
                       help='Only compare original and synthetic data (skip generation)')
    args = parser.parse_args()
    
    # Analyze original data
    print("Analyzing original data...")
    analysis = analyze_csv(input_file)
    
    # Process outliers sequentially (avoiding multiprocessing due to pickling issues)
    print("Detecting outliers in columns...")
    start_time = time.time()
    
    outliers = {}
    for col in analysis['numeric_columns']:
        col_start = time.time()
        
        # Calculate outlier statistics
        Q1 = analysis['dataframe'][col].quantile(0.25)
        Q3 = analysis['dataframe'][col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_count = ((analysis['dataframe'][col] < lower_bound) | 
                          (analysis['dataframe'][col] > upper_bound)).sum()
        outlier_percent = (outlier_count / len(analysis['dataframe'])) * 100
        
        elapsed = time.time() - col_start
        
        # Store results
        outliers[col] = {
            'count': outlier_count,
            'percent': outlier_percent,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
        print(f"  {col}: {outlier_count} outliers ({outlier_percent:.2f}%) - processed in {elapsed:.2f}s")
    
    total_elapsed = time.time() - start_time
    print(f"Outlier detection completed in {total_elapsed:.2f}s")
    
    analysis['outliers'] = outliers
    
    # Analyze distribution types for numeric columns (using fast method by default)
    print(f"Detecting distributions (using {'fast' if args.fast_distribution else 'detailed'} method)...")
    
    # Avoiding multiprocessing due to pickling issues
    total_start_time = time.time()
    
    distributions = {}
    for col in analysis['numeric_columns']:
        col_start = time.time()
        
        # Detect distribution 
        dist_type = detect_distribution(analysis['dataframe'][col], fast=args.fast_distribution)
        
        elapsed = time.time() - col_start
        distributions[col] = dist_type
        
        print(f"Column {col} best fits a {dist_type} distribution (processed in {elapsed:.2f}s)")
    
    total_elapsed = time.time() - total_start_time
    print(f"Distribution detection completed in {total_elapsed:.2f}s")
    
    analysis['distributions'] = distributions
    
    # Analyze time patterns if date column exists
    if analysis['date_columns'] and len(analysis['date_columns']) > 0:
        date_col = analysis['date_columns'][0]
        time_patterns = analyze_time_patterns(analysis['dataframe'], date_col)
        if time_patterns:
            analysis['time_patterns'] = time_patterns
    
    # Handle synthetic data
    if args.analyze_existing or args.compare_only:
        # Load existing synthetic data for analysis
        if os.path.exists(output_file):
            print(f"\nLoading existing synthetic data from {output_file}")
            synthetic_df = pd.read_csv(output_file)
            
            # Convert date columns to datetime
            if analysis['date_columns']:
                for col in analysis['date_columns']:
                    if col in synthetic_df.columns:
                        synthetic_df[col] = pd.to_datetime(synthetic_df[col])
        else:
            print(f"Error: {output_file} does not exist for analysis")
            if args.compare_only:
                print("Cannot continue with compare-only option")
                return
            args.analyze_existing = False  # Fall back to generating new data
    
    if not args.analyze_existing and not args.compare_only:
        # Generate new synthetic data
        print("\nGenerating synthetic data...")
        synthetic_df = generate_synthetic_data(analysis, start_date='2025-06-01', end_date='2025-06-06')
        
        # Save synthetic data
        print(f"\nSaving synthetic data to {output_file}")
        synthetic_df.to_csv(output_file, index=False)
        print("Done!")
    
    # Display sample
    print("\nSample of synthetic data:")
    print(synthetic_df.head())
    
    # Analyze and compare synthetic data with original
    comparison_report = analyze_and_compare(analysis['dataframe'], synthetic_df, analysis)
    
    # Generate some plots to compare original and synthetic data
    print("\nGenerating comparison plots...")
    
    # Use up to 3 most significant numeric columns for plotting
    if len(analysis['numeric_columns']) > 3:
        # Find columns with highest coefficient of variation (std/mean)
        cv_scores = {}
        for col in analysis['numeric_columns']:
            mean = analysis['numeric_stats'].loc[col, 'mean']
            std = analysis['numeric_stats'].loc[col, 'std']
            if mean != 0:
                cv_scores[col] = std / abs(mean)
            else:
                cv_scores[col] = 0
        
        # Sort by coefficient of variation
        sorted_cols = sorted(cv_scores.items(), key=lambda x: x[1], reverse=True)
        numeric_cols = [col for col, score in sorted_cols[:3]]
        print(f"Selected columns with highest variation for plotting: {numeric_cols}")
    else:
        numeric_cols = analysis['numeric_columns']
    
    # Add time-based analysis plots
    if 'time_patterns' in analysis and analysis['date_columns']:
        date_col = analysis['date_columns'][0]
        
        # Create day of week distribution plot
        plt.figure(figsize=(12, 6))
        
        # Add day of week to synthetic data
        synthetic_df[date_col] = pd.to_datetime(synthetic_df[date_col])
        synthetic_df['day_of_week'] = synthetic_df[date_col].dt.dayofweek
        
        # Group by day of week
        orig_dow = analysis['dataframe'].groupby('day_of_week').size().reindex(range(7), fill_value=0)
        synth_dow = synthetic_df.groupby('day_of_week').size().reindex(range(7), fill_value=0)
        
        # Normalize for comparison
        orig_dow = orig_dow / orig_dow.sum()
        synth_dow = synth_dow / synth_dow.sum()
        
        # Plot
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        plt.bar(np.arange(7) - 0.2, orig_dow, width=0.4, label='Original', alpha=0.7)
        plt.bar(np.arange(7) + 0.2, synth_dow, width=0.4, label='Synthetic', alpha=0.7)
        plt.xticks(np.arange(7), days, rotation=45)
        plt.title('Day of Week Distribution')
        plt.legend()
        plt.tight_layout()
        plt.savefig('figures/day_of_week_comparison.png')
        plt.close()
    
    # Plot histograms to compare distributions
    for col in numeric_cols:
        plt.figure(figsize=(10, 6))
        
        # Plot original data
        plt.hist(analysis['dataframe'][col], alpha=0.5, bins=30, label='Original')
        
        # Plot synthetic data
        plt.hist(synthetic_df[col], alpha=0.5, bins=30, label='Synthetic')
        
        plt.title(f'Distribution Comparison for {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.legend()
        
        # Save figure
        plt.savefig(f'figures/{col}_comparison.png')
        plt.close()
    
    print("Plots saved to 'figures/' directory")

    # Create Q-Q plots to compare distributions
    try:
        import scipy.stats as stats
        for col in numeric_cols:
            plt.figure(figsize=(10, 6))
            
            # Get data
            orig_data = analysis['dataframe'][col].dropna()
            synth_data = synthetic_df[col].dropna()
            
            # Sample if too many points
            if len(orig_data) > 1000:
                orig_data = orig_data.sample(1000)
            if len(synth_data) > 1000:
                synth_data = synth_data.sample(1000)
            
            # Create QQ plot
            stats.probplot(orig_data, dist="norm", plot=plt)
            plt.title(f'Q-Q Plot for {col} (Original Data)')
            plt.savefig(f'figures/{col}_qq_original.png')
            plt.close()
            
            plt.figure(figsize=(10, 6))
            stats.probplot(synth_data, dist="norm", plot=plt)
            plt.title(f'Q-Q Plot for {col} (Synthetic Data)')
            plt.savefig(f'figures/{col}_qq_synthetic.png')
            plt.close()
    except Exception as e:
        print(f"Error creating Q-Q plots: {e}")
    
    # Plot histograms to compare distributions
    for col in numeric_cols:
        plt.figure(figsize=(10, 6))
        
        # Plot original data
        plt.hist(analysis['dataframe'][col], alpha=0.5, bins=30, label='Original')
        
        # Plot synthetic data
        plt.hist(synthetic_df[col], alpha=0.5, bins=30, label='Synthetic')
        
        plt.title(f'Distribution Comparison for {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.legend()
        
        # Save figure
        plt.savefig(f'figures/{col}_comparison.png')
        plt.close()
    
    # Create boxplots to show outliers
    for col in numeric_cols:
        plt.figure(figsize=(10, 6))
        plt.boxplot([analysis['dataframe'][col].dropna(), synthetic_df[col].dropna()], 
                   labels=['Original', 'Synthetic'])
        plt.title(f'Boxplot for {col}')
        plt.savefig(f'figures/{col}_boxplot.png')
        plt.close()
    
    print("Plots saved to 'figures/' directory")
    
    # Generate summary report
    with open('synthetic_data_report.txt', 'w') as f:
        f.write("Synthetic Data Generation Report\n")
        f.write("==============================\n\n")
        f.write(f"Original file: {input_file}\n")
        f.write(f"Output file: {output_file}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Original data shape: {analysis['dataframe'].shape}\n")
        f.write(f"Synthetic data shape: {synthetic_df.shape}\n\n")
        f.write("Numeric column statistics (Original vs. Synthetic):\n")
        
        for col in numeric_cols:
            f.write(f"\n{col}:\n")
            orig_stats = analysis['dataframe'][col].describe()
            synth_stats = synthetic_df[col].describe()
            
            f.write(f"  Mean: {orig_stats['mean']:.2f} vs {synth_stats['mean']:.2f}\n")
            f.write(f"  Std: {orig_stats['std']:.2f} vs {synth_stats['std']:.2f}\n")
            f.write(f"  Min: {orig_stats['min']:.2f} vs {synth_stats['min']:.2f}\n")
            f.write(f"  Max: {orig_stats['max']:.2f} vs {synth_stats['max']:.2f}\n")
            
            if col in analysis['distributions']:
                f.write(f"  Distribution: {analysis['distributions'][col]}\n")
            
            if col in analysis['outliers']:
                f.write(f"  Original outliers: {analysis['outliers'][col]['count']} ({analysis['outliers'][col]['percent']:.2f}%)\n")
        
        f.write("\nPlots generated in 'figures/' directory\n")
    
    print("Generated summary report: synthetic_data_report.txt")

if __name__ == "__main__":
    main()
