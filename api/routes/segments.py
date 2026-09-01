# api/routes/segments.py
"""
GET /api/segments
Returns overview of all customer segments with
sizes and average RFM values.
"""

from fastapi import APIRouter, Request
from src.features import FEATURES

router = APIRouter()

SEGMENT_DESCRIPTIONS = {
    "Champions":        "Highest spenders who buy frequently and recently.",
    "Occasional Buyers":"Mid-tier customers who buy sometimes but not regularly.",
    "At-Risk Customers":"Customers who haven't purchased recently and may churn."
}


@router.get("/segments")
def get_segments(request: Request):
    """
    Returns all customer segments with:
    - Segment name and description
    - Number of customers in each segment
    - Average RFM feature values per segment
    """
    customer_data = request.app.state.customer_data

    # Group by segment and compute stats
    grouped = customer_data.groupby("Segment_Name")

    segments = []
    for segment_name, group in grouped:
        avg_rfm = group[FEATURES].mean().round(2)
        segments.append({
            "name":                       segment_name,
            "description":                SEGMENT_DESCRIPTIONS.get(segment_name, ""),
            "size":                       len(group),
            "avg_days_since_purchase":    avg_rfm["Days_Since_Last_Purchase"],
            "avg_transactions":           avg_rfm["Total_Transactions"],
            "avg_products_purchased":     avg_rfm["Total_Products_Purchased"],
            "avg_total_spend":            avg_rfm["Total_Spend"],
            "avg_transaction_value":      avg_rfm["Average_Transaction_Value"],
        })

    # Sort by avg spend descending
    segments.sort(key=lambda x: x["avg_total_spend"], reverse=True)

    return {
        "total_customers": len(customer_data),
        "total_segments":  len(segments),
        "segments":        segments
    }