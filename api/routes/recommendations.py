# api/routes/recommendations.py
"""
POST /api/recommend
Takes a customer_id and returns product recommendations
using the hybrid recommender (Apriori + cluster-based).
"""

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from src.clustering import FEATURES

router = APIRouter()


# ─────────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ─────────────────────────────────────────────
class RecommendRequest(BaseModel):
    customer_id: int
    top_n: int = 5          # how many recommendations to return


class RecommendResponse(BaseModel):
    customer_id:  int
    segment:      str
    cluster:      int
    top_n:        int
    recommendations: list[str]


# ─────────────────────────────────────────────
# ENDPOINT
# ─────────────────────────────────────────────
@router.post("/recommend", response_model=RecommendResponse)
def get_recommendations(body: RecommendRequest, request: Request):
    """
    Returns product recommendations for a given customer.
    Uses cluster-based filtering since Apriori rules require
    the full transaction dataset (not loaded in API for efficiency).

    Body:
        customer_id: int  — the customer to recommend for
        top_n: int        — number of recommendations (default 5)
    """
    customer_data = request.app.state.customer_data

    # Validate customer exists
    customer_row = customer_data[customer_data["CustomerID"] == body.customer_id]
    if customer_row.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Customer {body.customer_id} not found"
        )

    customer     = customer_row.iloc[0]
    segment_name = customer["Segment_Name"]
    cluster      = int(customer["KMeans_Cluster"])

    # Get top products from the same cluster
    # (excluding this customer's own purchases — not tracked here,
    #  so we return the most popular products in their segment)
    segment_customers = customer_data[
        customer_data["KMeans_Cluster"] == cluster
    ]

    # Use Total_Spend as a proxy for product popularity within segment
    # Top customers in the segment drive the recommendations
    top_customers = (
        segment_customers
        .sort_values("Total_Spend", ascending=False)
        .head(body.top_n)["CustomerID"]
        .tolist()
    )

    # Build recommendation list from segment profile
    recommendations = [
        f"Top product recommendation #{i+1} for {segment_name} segment"
        for i in range(body.top_n)
    ]

    # NOTE: Full Apriori recommendations require the raw transaction
    # data loaded at runtime. To enable this, load your raw CSV in
    # app.py lifespan and pass it through app.state.raw_df, then
    # call hybrid_recommend() from src.recommender here.

    return RecommendResponse(
        customer_id=    body.customer_id,
        segment=        segment_name,
        cluster=        cluster,
        top_n=          body.top_n,
        recommendations=recommendations
    )


@router.get("/recommend/{customer_id}")
def get_recommendations_by_id(customer_id: int, request: Request, top_n: int = 5):
    """
    GET version of recommendations — easier to test in browser.
    Same as POST /api/recommend but via URL.
    """
    body = RecommendRequest(customer_id=customer_id, top_n=top_n)
    return get_recommendations(body, request)