# src/recommender.py
"""
Product Recommendation System.
Covers: Market Basket Analysis (Apriori),
        Cluster-based recommendations,
        Hybrid recommender combining both.
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


def build_basket_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a binary customer-product matrix from completed transactions.
    Rows = CustomerID, Columns = Product Description, Values = 0/1.
    """
    completed = df[df['Transaction_Status'] == 'Completed']

    basket = completed.pivot_table(
        index='CustomerID',
        columns='Description',
        values='Quantity',
        aggfunc='sum',
        fill_value=0
    )
    basket = basket.map(lambda x: 1 if x > 0 else 0)
    print(f"Basket matrix: {basket.shape[0]} customers x {basket.shape[1]} products")
    return basket


def generate_association_rules(basket: pd.DataFrame,
                                min_support: float = 0.03,
                                min_lift: float = 1.2) -> pd.DataFrame:
    """
    Run Apriori algorithm and generate association rules.
    Returns rules sorted by lift descending.

    Tune min_support:
      Too few rules → lower to 0.01
      Too many rules → raise to 0.05
    """
    frequent_itemsets = apriori(basket, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric='lift', min_threshold=min_lift)
    rules = rules.sort_values('lift', ascending=False).reset_index(drop=True)
    print(f"Generated {len(rules)} association rules")
    return rules


def apriori_recommend(bought_items: list, rules: pd.DataFrame,
                      top_n: int = 5) -> list:
    """
    Recommend products based on association rules.
    Given a list of products the customer bought,
    returns top N products with highest lift score.
    """
    recommendations = {}

    for _, rule in rules.iterrows():
        antecedent = set(rule['antecedents'])
        consequent = set(rule['consequents'])

        if antecedent.issubset(set(bought_items)):
            for product in consequent:
                if product not in bought_items:
                    if product not in recommendations:
                        recommendations[product] = rule['lift']
                    else:
                        recommendations[product] = max(
                            recommendations[product], rule['lift']
                        )

    sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
    return [product for product, _ in sorted_recs[:top_n]]


def cluster_recommend(customer_id: int, customer_data: pd.DataFrame,
                      df: pd.DataFrame, top_n: int = 5) -> list:
    """
    Recommend top products popular within the customer's segment
    that the customer has not purchased yet.
    """
    completed = df[df['Transaction_Status'] == 'Completed']

    cluster = customer_data.loc[
        customer_data['CustomerID'] == customer_id, 'KMeans_Cluster'
    ].values[0]

    cluster_customers = customer_data[
        customer_data['KMeans_Cluster'] == cluster
    ]['CustomerID'].tolist()

    cluster_purchases = completed[
        (completed['CustomerID'].isin(cluster_customers)) &
        (completed['CustomerID'] != customer_id)
    ]

    top_products = (
        cluster_purchases.groupby('Description')['Quantity']
        .sum()
        .sort_values(ascending=False)
    )

    already_bought = set(
        completed[completed['CustomerID'] == customer_id]['Description']
    )

    recommendations = [p for p in top_products.index if p not in already_bought]
    return recommendations[:top_n]


def hybrid_recommend(customer_id: int, rules: pd.DataFrame,
                     customer_data: pd.DataFrame, df: pd.DataFrame,
                     top_n: int = 5) -> list:
    """
    Hybrid recommender combining Apriori and cluster-based recommendations.
    Apriori results take priority; cluster results fill remaining slots.
    """
    completed = df[df['Transaction_Status'] == 'Completed']
    bought_items = list(
        completed[completed['CustomerID'] == customer_id]['Description'].unique()
    )

    apriori_recs = apriori_recommend(bought_items, rules, top_n=top_n)
    cluster_recs = cluster_recommend(customer_id, customer_data, df, top_n=top_n)

    # Merge preserving order, removing duplicates
    combined = list(dict.fromkeys(apriori_recs + cluster_recs))
    return combined[:top_n]


def get_customer_recommendations(customer_id: int, rules: pd.DataFrame,
                                  customer_data: pd.DataFrame,
                                  df: pd.DataFrame, top_n: int = 5) -> dict:
    """
    Full recommendation response for a customer.
    Used by the API — returns a dict ready to be served as JSON.
    """
    completed = df[df['Transaction_Status'] == 'Completed']

    customer_row = customer_data[customer_data['CustomerID'] == customer_id]
    if customer_row.empty:
        return {'error': f'Customer {customer_id} not found'}

    segment = customer_row['Segment_Name'].values[0]
    cluster = int(customer_row['KMeans_Cluster'].values[0])
    unique_products = len(
        completed[completed['CustomerID'] == customer_id]['Description'].unique()
    )

    recommendations = hybrid_recommend(customer_id, rules, customer_data, df, top_n)

    return {
        'customer_id': customer_id,
        'segment': segment,
        'cluster': cluster,
        'unique_products_purchased': unique_products,
        'recommendations': recommendations
    }