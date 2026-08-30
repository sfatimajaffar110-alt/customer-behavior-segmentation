from src.preprocessing import clean_pipeline
from src.features import build_rfm_features, apply_iqr_capping
from src.clustering import scale_features, train_kmeans, evaluate_clustering
from src.recommender import get_customer_recommendations

print("All imports successful ✅")