# customer-behavior-segmentation


## Overview
Unsupervised machine learning project to segment customers based on 
purchasing behavior using RFM analysis, K-Means, and Hierarchical Clustering.

## Dataset
Online retail transaction data containing invoices, products, quantities, 
prices, and customer IDs.
Download the dataset from Kaggle:
https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci
Place it in the root folder and rename it to `data.csv`

## How to Run
1. Clone the repo
2. Install dependencies: pip install -r requirements.txt
3. Open the notebook: customer_behavior_segmentation.ipynb

## Techniques Used
- RFM Feature Engineering
- K-Means Clustering
- Hierarchical Clustering
- Silhouette Score Evaluation

## Future Work
- Outlier treatment
- DBSCAN clustering
- Product Recommendation System

## Progress
- ✅ Phase 1: EDA, Cleaning, RFM Features, Outlier Treatment, 
              Clustering, PCA Visualization, Cluster Profiling
- 🔄 Phase 2: Modular src/ files (in progress)
- ⏳ Phase 3: FastAPI Backend
- ⏳ Phase 4: Dashboard
