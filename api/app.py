"""
FastAPI entry point.
Loads data and models once at startup, then shares them
across all routes via app.state.
"""

import pandas as pd
import joblib
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from api.routes import segments, customers, recommendations


# ─────────────────────────────────────────────
# STARTUP — load everything once when server starts
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and data on startup, clean up on shutdown"""
    print("Loading models and data...")

    #load saved models
    app.state.kmeans = joblib.load("outputs/models/kmeans_model.joblib")
    app.state.scaler = joblib.load("outputs/models/scaler.joblib")

    #Load processed customr data
    app.state.customer_data = pd.read_csv("dataset/processed/customer_rfm.csv")
    app.state.customer_data["CustomerID"] = app.state.customer_data["CustomerID"].astype(int)

    print(f"Loaded {len(app.state.customer_data)} customer")
    print("API ready!")

    yield #server runs here

    print("shutting down...")
# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────
app = FastAPI(
    title="Customer Segmentation API",
    description="API for customer behavior segmentation and product recommendations",
    version="1.0.0",
    lifespan=lifespan
)
 
# Allow the dashboard (frontend) to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in production, replace * with your dashboard URL
    allow_methods=["*"],
    allow_headers=["*"],
)
 
# Register routes
app.include_router(segments.router,        prefix="/api")
app.include_router(customers.router,       prefix="/api")
app.include_router(recommendations.router, prefix="/api")
 
 
# ─────────────────────────────────────────────
# ROOT ENDPOINT
# ─────────────────────────────────────────────
@app.get("/debug")
def debug(request: Request):
    df = request.app.state.customer_data
    return {
        "total_rows": len(df),
        "columns": df.columns.tolist(),
        "customer_id_dtype": str(df["CustomerID"].dtype),
        "sample_ids": df["CustomerID"].head(5).tolist(),
        "search_12347": len(df[df["CustomerID"] == 12347])
    }

@app.get("/")
def root():
    return {
        "message": "Customer Segmentation API is running",
        "docs": "/docs",
        "endpoints": [
            "GET  /api/segments",
            "GET  /api/customer/{customer_id}",
            "POST /api/recommend"
        ]
    }