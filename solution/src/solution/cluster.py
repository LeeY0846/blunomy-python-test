import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import HDBSCAN

def cluster_hdbscan(data: pd.DataFrame):
    data = data.copy()

    pca_result = PCA(n_components=3).fit_transform(data)

    data["pc1"] = pca_result[:,0]
    data["pc2"] = pca_result[:,1]
    data["pc3"] = pca_result[:,2]

    X_cluster = np.column_stack([
        0.05 * data["pc1"],
        data["pc2"],
        data["pc3"],
    ])
    
    clustering = HDBSCAN(copy=True, min_cluster_size=20).fit(X_cluster)
    return clustering.labels_

def cluster_wires(data: pd.DataFrame, cluster_method: str):
    if cluster_method != "hdbscan":
        raise AttributeError(f"Unknown cluster method - {cluster_method}")
    # TODO: Add more cluster methods for broader scenarios

    data = data.copy()
    labels = cluster_hdbscan(data)

    data["label"] = labels

    valid_labels = sorted(label for label in np.unique(data["label"]) if label >= 0)
    return [data[data["label"] == label].drop(columns=["label"]).copy() for label in valid_labels]