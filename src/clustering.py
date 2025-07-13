from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import pandas as pd

def run_clustering():
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = {}

    kmeans = KMeans(n_clusters=3, random_state=42)
    results['KMeans'] = kmeans.fit_predict(X_scaled)

    dbscan = DBSCAN(eps=0.5, min_samples=5)
    results['DBSCAN'] = dbscan.fit_predict(X_scaled)

    agglomerative = AgglomerativeClustering(n_clusters=3)
    results['Hierarchical'] = agglomerative.fit_predict(X_scaled)

    gmm = GaussianMixture(n_components=3, random_state=42)
    results['GMM'] = gmm.fit_predict(X_scaled)

    return results
