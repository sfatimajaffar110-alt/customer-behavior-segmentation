"""
data loading and cleaning functions
covers: null handling, duplicaes, cancelled transactions, stokcode anomalies, description cleaning, zero prices.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
 
 
def load_data(path: str) -> pd.DataFrame:
    """Load raw CSV dataset."""
    df = pd.read_csv(path, encoding='latin-1')
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df
 
 
def plot_missing_values(df: pd.DataFrame, save_path: str = None) -> None:
    """Plot horizontal bar chart of missing value percentages."""
    missing_data = df.isnull().sum()
    missing_percentage = (missing_data[missing_data > 0] / df.shape[0]) * 100
    missing_percentage.sort_values(ascending=True, inplace=True)
 
    fig, ax = plt.subplots(figsize=(15, 4))
    ax.barh(missing_percentage.index, missing_percentage, color='blue')
    for i, (value, name) in enumerate(zip(missing_percentage, missing_percentage.index)):
        ax.text(value + 0.5, i, f"{value:.2f}%", ha='left', va='center',
                fontweight='bold', fontsize=18, color='black')
    ax.set_xlim([0, 40])
    plt.title("Percentage of Missing Values", fontweight='bold', fontsize=22)
    plt.xlabel('Percentages (%)', fontsize=16)
    plt.tight_layout()
 
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()
 
 
def remove_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where CustomerID or Description is null."""
    before = df.shape[0]
    df = df.dropna(subset=['CustomerID', 'Description'])
    print(f"Removed {before - df.shape[0]} rows with null CustomerID/Description")
    return df
 
 
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows."""
    before = df.shape[0]
    df.drop_duplicates(inplace=True)
    print(f"Removed {before - df.shape[0]} duplicate rows")
    return df
 
 
def flag_cancelled_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Add Transaction_Status column: Cancelled or Completed."""
    df['Transaction_Status'] = np.where(
        df['InvoiceNo'].astype(str).str.startswith('C'), 'Cancelled', 'Completed'
    )
    cancelled = (df['Transaction_Status'] == 'Cancelled').sum()
    print(f"Flagged {cancelled} cancelled transactions")
    return df
 
 
def remove_stockcode_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with non-product stock codes (0 or 1 numeric characters)."""
    unique_codes = df['StockCode'].unique()
    anomalous = [
        code for code in unique_codes
        if sum(c.isdigit() for c in str(code)) in (0, 1)
    ]
    before = df.shape[0]
    df = df[~df['StockCode'].isin(anomalous)]
    print(f"Removed {before - df.shape[0]} rows with anomalous stock codes")
    return df
 
 
def clean_descriptions(df: pd.DataFrame) -> pd.DataFrame:
    """Remove service-related descriptions and standardize to uppercase."""
    service_descriptions = ["Next Day Carriage", "High Resolution Image"]
    before = df.shape[0]
    df = df[~df['Description'].isin(service_descriptions)]
    df['Description'] = df['Description'].str.upper()
    print(f"Removed {before - df.shape[0]} service-related description rows")
    return df
 
 
def remove_zero_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where UnitPrice is zero."""
    before = df.shape[0]
    df = df[df['UnitPrice'] > 0]
    print(f"Removed {before - df.shape[0]} rows with zero unit price")
    return df
 
 
def clean_pipeline(path: str, figures_dir: str = 'outputs/figures') -> pd.DataFrame:
    """
    Run the full cleaning pipeline in one call.
    Returns a clean DataFrame ready for feature engineering.
    """
    df = load_data(path)
    plot_missing_values(df, save_path=f'{figures_dir}/missing_values.png')
    df = remove_nulls(df)
    df = remove_duplicates(df)
    df = flag_cancelled_transactions(df)
    df = remove_stockcode_anomalies(df)
    df = clean_descriptions(df)
    df = remove_zero_prices(df)
    df.reset_index(drop=True, inplace=True)
    print(f"\nClean dataset: {df.shape[0]} rows remaining")
    return df