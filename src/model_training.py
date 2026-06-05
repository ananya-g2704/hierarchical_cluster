import joblib

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from data_preprocessing import (
    load_data,
    preprocess_data
)

df = load_data()

X_scaled, scaler = preprocess_data(df)

model = AgglomerativeClustering(
    n_clusters=4,
    linkage="ward"
)

clusters = model.fit_predict(
    X_scaled
)

score = silhouette_score(
    X_scaled,
    clusters
)

print("="*50)
print("Hierarchical Clustering Training Complete")
print(f"Silhouette Score : {score:.3f}")
print("="*50)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

print("Scaler Saved")