# api/routes/customers.py
"""
GET /api/customer/{customer_id}
Returns a single customer's segment and full RFM profile.
"""

# api/routes/customers.py
from fastapi import APIRouter, Request, HTTPException

router = APIRouter()

FEATURES = [
    'Days_Since_Last_Purchase',
    'Total_Transactions',
    'Total_Products_Purchased',
    'Total_Spend',
    'Average_Transaction_Value'
]


@router.get("/customer/{customer_id}")
def get_customer(customer_id: int, request: Request):
    customer_data = request.app.state.customer_data

    customer_row = customer_data[customer_data["CustomerID"] == customer_id]
    if customer_row.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Customer {customer_id} not found"
        )

    customer = customer_row.iloc[0]
    segment_name = customer["Segment_Name"]

    segment_avg = (
        customer_data[customer_data["Segment_Name"] == segment_name][FEATURES]
        .mean()
        .round(2)
    )

    return {
        "customer_id":            customer_id,
        "segment":                segment_name,
        "cluster":                int(customer["KMeans_Cluster"]),
        "rfm": {
            "days_since_last_purchase":  round(float(customer["Days_Since_Last_Purchase"]), 2),
            "total_transactions":        round(float(customer["Total_Transactions"]), 2),
            "total_products_purchased":  round(float(customer["Total_Products_Purchased"]), 2),
            "total_spend":               round(float(customer["Total_Spend"]), 2),
            "average_transaction_value": round(float(customer["Average_Transaction_Value"]), 2),
        },
        "segment_averages": {
            "days_since_last_purchase":  float(segment_avg["Days_Since_Last_Purchase"]),
            "total_transactions":        float(segment_avg["Total_Transactions"]),
            "total_products_purchased":  float(segment_avg["Total_Products_Purchased"]),
            "total_spend":               float(segment_avg["Total_Spend"]),
            "average_transaction_value": float(segment_avg["Average_Transaction_Value"]),
        }
    }


@router.get("/customers")
def get_all_customers(request: Request, limit: int = 50, offset: int = 0):
    customer_data = request.app.state.customer_data
    total = len(customer_data)
    page = customer_data.iloc[offset: offset + limit]

    customers = []
    for _, row in page.iterrows():
        customers.append({
            "customer_id":        int(row["CustomerID"]),
            "segment":            row["Segment_Name"],
            "cluster":            int(row["KMeans_Cluster"]),
            "total_spend":        round(float(row["Total_Spend"]), 2),
            "total_transactions": int(row["Total_Transactions"]),
        })

    return {
        "total":     total,
        "limit":     limit,
        "offset":    offset,
        "customers": customers
    }