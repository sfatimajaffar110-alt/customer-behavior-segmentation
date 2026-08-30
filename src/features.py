# src/features.py
"""
RFM Feature Engineering functions.
Covers: Recency, Frequency, Monetary calculation,
        outlier treatment via IQR capping,
        distribution plots before/after treatment.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


FEATURES = [
    'Days_Since_Last_Purchase',
    'Total_Transactions',
    'Total_Products_Purchased',
    'Total_Spend',
    'Average_Transaction_Value'
]


def compute_recency(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate days since last purchase per customer."""
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['InvoiceDay'] = df['InvoiceDate'].dt.date

    customer_data = df.groupby('CustomerID')['InvoiceDay'].max().reset_index()
    most_recent_date = pd.to_datetime(df['InvoiceDay'].max())
    customer_data['InvoiceDay'] = pd.to_datetime(customer_data['InvoiceDay'])
    customer_data['Days_Since_Last_Purchase'] = (
        most_recent_date - customer_data['InvoiceDay']
    ).dt.days
    customer_data.drop(columns=['InvoiceDay'], inplace=True)
    return customer_data


def compute_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate total transactions and total products purchased per customer."""
    total_transactions = df.groupby('CustomerID')['InvoiceNo'].nunique().reset_index()
    total_transactions.rename(columns={'InvoiceNo': 'Total_Transactions'}, inplace=True)

    total_products = df.groupby('CustomerID')['Quantity'].sum().reset_index()
    total_products.rename(columns={'Quantity': 'Total_Products_Purchased'}, inplace=True)

    frequency_df = pd.merge(total_transactions, total_products, on='CustomerID')
    return frequency_df


def compute_monetary(df: pd.DataFrame, total_transactions: pd.DataFrame) -> tuple:
    """Calculate total spend and average transaction value per customer."""
    df['Total_Spend'] = df['UnitPrice'] * df['Quantity']
    total_spend = df.groupby('CustomerID')['Total_Spend'].sum().reset_index()

    avg_transaction = total_spend.merge(total_transactions, on='CustomerID')
    avg_transaction['Average_Transaction_Value'] = (
        avg_transaction['Total_Spend'] / avg_transaction['Total_Transactions']
    )
    return total_spend, avg_transaction[['CustomerID', 'Average_Transaction_Value']]


def build_rfm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine Recency, Frequency, and Monetary into one customer DataFrame.
    Returns the full RFM feature table indexed by CustomerID.
    """
    customer_data = compute_recency(df)

    frequency_df = compute_frequency(df)
    customer_data = pd.merge(customer_data, frequency_df, on='CustomerID')

    total_transactions = frequency_df[['CustomerID', 'Total_Transactions']]
    total_spend, avg_transaction = compute_monetary(df, total_transactions)
    customer_data = pd.merge(customer_data, total_spend, on='CustomerID')
    customer_data = pd.merge(customer_data, avg_transaction, on='CustomerID')

    print(f"RFM features built for {len(customer_data)} customers")
    print(customer_data[FEATURES].describe().round(2))
    return customer_data


def plot_distributions(customer_data: pd.DataFrame, stage: str = 'Before',
                       color: str = 'steelblue', save_path: str = None) -> None:
    """Plot histogram distributions of all RFM features."""
    fig, axes = plt.subplots(1, len(FEATURES), figsize=(20, 4))
    for ax, col in zip(axes, FEATURES):
        ax.hist(customer_data[col], bins=50, color=color, edgecolor='white')
        ax.set_title(f'{col}\n({stage})', fontsize=9)
        ax.set_xlabel('Value')
    plt.suptitle(
        f'RFM Feature Distributions — {stage} Outlier Treatment',
        fontweight='bold'
    )
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()


def apply_iqr_capping(customer_data: pd.DataFrame,
                      figures_dir: str = 'outputs/figures') -> pd.DataFrame:
    """
    Cap outliers in all RFM features using IQR method.
    Plots distributions before and after treatment.
    Returns cleaned copy of customer_data.
    """
    plot_distributions(
        customer_data, stage='Before', color='steelblue',
        save_path=f'{figures_dir}/distributions_before.png'
    )

    customer_data_clean = customer_data.copy()
    for col in FEATURES:
        Q1 = customer_data_clean[col].quantile(0.25)
        Q3 = customer_data_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        before_max = customer_data_clean[col].max()
        customer_data_clean[col] = customer_data_clean[col].clip(lower, upper)
        after_max = customer_data_clean[col].max()
        print(f"{col}: max {before_max:.1f} → {after_max:.1f}")

    plot_distributions(
        customer_data_clean, stage='After', color='seagreen',
        save_path=f'{figures_dir}/distributions_after.png'
    )
    return customer_data_clean