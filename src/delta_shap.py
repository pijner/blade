# detect_backdoor_shap.py

import shap
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


def train_rf(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model


class DeltaSHAP:
    def __init__(self):
        pass

    @staticmethod
    def get_shap_values(model, X):
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X)
        if shap_vals.ndim == 3:  # multi-class
            shap_vals = np.mean(np.abs(shap_vals), axis=-1)
        return shap_vals

    @staticmethod
    def compute_deltas(model_a, model_b, X_combined):
        shap_a = DeltaSHAP.get_shap_values(model_a, X_combined)
        shap_b = DeltaSHAP.get_shap_values(model_b, X_combined)
        delta = shap_b - shap_a
        return delta

    @staticmethod
    def cluster_deltas(delta_shap, eps=1.5, min_samples=5):
        delta_scaled = StandardScaler().fit_transform(delta_shap)
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(delta_scaled)
        return clustering.labels_

    @staticmethod
    def visualize_clusters(delta_shap, labels):
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(delta_shap)
        df = pd.DataFrame(reduced, columns=["PC1", "PC2"])
        df["Cluster"] = labels

        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x="PC1", y="PC2", hue="Cluster", palette="tab10")
        plt.title("DBSCAN Clustering of ΔSHAP Values")
        plt.savefig("delta_shap_clusters.jpg")

    @staticmethod
    def get_sus_indices(delta_shap, n=50):
        delta_norm = np.linalg.norm(delta_shap, axis=1)
        suspicious_indices = np.argsort(-delta_norm)[:n]
        return suspicious_indices


def run_detection(X_trusted, y_trusted, X_full, y_full, N=50):
    model_a = train_rf(X_trusted, y_trusted)
    model_b = train_rf(X_full, y_full)

    delta_shap = DeltaSHAP.compute_deltas(model_a, model_b, X_full)
    suspicious_indices = DeltaSHAP.get_sus_indices(delta_shap, N)

    print(f"Top {N} suspicious indices based on ΔSHAP values: {suspicious_indices}")


# --- Example usage on dummy data ---
if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    # Simulate trusted/untrusted split
    X_trusted = X.sample(frac=0.1, random_state=42)
    y_trusted = y.loc[X_trusted.index]

    X_combined = X
    y_combined = y

    run_detection(X_trusted, y_trusted, X_combined, y_combined)
