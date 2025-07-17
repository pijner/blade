# detect_backdoor_shap.py

import logging
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from src.model_factory import set_seed
from src.pre_processing import GenericPreProcessor
from src.model_training import ModelTrainer, load_data, train_models, load_models, prepare_data_for_training
from src.backdoor_attack import BackdoorPoisoner
from src.util import load_yaml


def train_rf(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model


class DeltaSHAP:
    def __init__(self):
        pass

    @staticmethod
    def compute_shap_values_batched(explainer, X_to_explain, batch_size=64):
        shap_vals = []
        num_samples = X_to_explain[0].shape[0] if isinstance(X_to_explain, list) else X_to_explain.shape[0]
        for i in range(0, num_samples, batch_size):
            batch = (
                [_x[i : i + batch_size] for _x in X_to_explain]
                if isinstance(X_to_explain, list)
                else X_to_explain[i : i + batch_size]
            )
            with tf.device("/cpu:0"):
                vals = explainer.shap_values(batch)
            if isinstance(vals, list):  # multi-class
                vals = np.mean(np.abs(vals), axis=0)
            shap_vals.append(vals)
        return np.concatenate(shap_vals, axis=0)

    @staticmethod
    def get_shap_values(model: ModelTrainer, X: pd.DataFrame):
        if model.model_type in ["rf", "xgb"]:
            explainer = shap.TreeExplainer(model.model)
        elif model.model_type in ["tf_mlp", "tabnet"]:
            background = train_test_split(X, test_size=100, random_state=42)[1]
            if model.model_type == "tf_mlp":
                background = model.model.make_inputs(*model.prepare_inputs(background))
                X = model.model.make_inputs(*model.prepare_inputs(X))
                explainer = shap.KernelExplainer(model.model.model, background)
            else:
                explainer = shap.GradientExplainer(model.model, background)

        # try predicting for sanity
        shap_vals = DeltaSHAP.compute_shap_values_batched(explainer, X, batch_size=100000)

        if shap_vals.ndim == 3:  # multi-class
            shap_vals = np.mean(np.abs(shap_vals), axis=-1)
        return shap_vals.astype(np.float32)

    @staticmethod
    def compute_deltas(model_a, model_b, X_combined: pd.DataFrame):
        shap_a = DeltaSHAP.get_shap_values(model_a, X_combined)
        shap_b = DeltaSHAP.get_shap_values(model_b, X_combined)
        delta = shap_b - shap_a
        return delta

    @staticmethod
    def cluster_deltas(delta_shap, eps: float = 1.5, min_samples: int = 5):
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
    def fast_cosine_distance(X1: np.ndarray, X2: np.ndarray, batch_size: int = 1000000) -> np.ndarray:
        """
        Computes cosine distance between corresponding rows of X1 and X2.
        Assumes X1 and X2 are both of shape (n_samples, n_features).
        Returns a vector of shape (n_samples,)
        """
        assert X1.shape == X2.shape
        n = X1.shape[0]
        result = np.empty(n)

        for i in range(0, n, batch_size):
            j = min(i + batch_size, n)
            A = X1[i:j]
            B = X2[i:j]

            # Normalize rows manually
            A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
            B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)

            # Compute 1 - cosine similarity
            result[i:j] = 1 - np.sum(A_norm * B_norm, axis=1)

        return result

    @staticmethod
    def get_sus_indices(
        shap_a, shap_b, n: int = 50, top_k: int = 5, alpha: float = 0.4, beta: float = 0.3, gamma: float = 0.3
    ):
        if alpha + beta + gamma != 1:
            logging.warning(
                f"Alpha ({alpha}), beta ({beta}), and gamma ({gamma}) should sum to 1 (not {alpha + beta + gamma}). Normalizing them."
            )
            total = alpha + beta + gamma
            alpha /= total
            beta /= total
            gamma /= total

        delta_shap = shap_b - shap_a

        # L2 norm
        l2_norm = np.linalg.norm(delta_shap, axis=1)

        # Top-k feature shifts
        topk_mean = np.sort(np.abs(delta_shap), axis=1)[:, -top_k:].mean(axis=1)

        # Cosine distance
        cosine_dist = DeltaSHAP.fast_cosine_distance(shap_a, shap_b)

        # Hybrid suspiciousness score
        score = alpha * l2_norm + beta * topk_mean + gamma * cosine_dist
        sus_indices = np.argsort(-score)[:n]

        return sus_indices


def run_detection(model_a: ModelTrainer, model_b: ModelTrainer, X_full: pd.DataFrame, N: int = 50):
    shap_a = DeltaSHAP.get_shap_values(model_a, X_full)
    shap_b = DeltaSHAP.get_shap_values(model_b, X_full)

    suspicious_indices = DeltaSHAP.get_sus_indices(shap_a, shap_b, N)
    return suspicious_indices


def train_trusted_and_untrusted_models(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, config_dict: dict
):
    trust_fraction = config_dict["trust_fraction"]
    trusted_indices = X_train.sample(frac=trust_fraction, random_state=42).index
    X_trusted = X_train.loc[trusted_indices]
    y_trusted = y_train.loc[trusted_indices]

    X_untrusted = X_train.drop(trusted_indices)
    y_untrusted = y_train.drop(trusted_indices)

    save_dir = Path(config_dict["save_dir"])
    trusted_models_dir = save_dir / "trusted_models"
    untrusted_models_dir = save_dir / "untrusted_models"

    # update save directory in config_dict
    config_dict["save_dir"] = trusted_models_dir
    # Prepare trusted data for training
    X_trusted_norm, y_trusted, X_test_norm, _ = prepare_data_for_training(X_trusted, y_trusted, X_test, config_dict)
    # Train models on trusted data
    logging.info("Training models on trusted data.")
    train_models(X_trusted_norm, y_trusted, X_test_norm, y_test, config_dict=config_dict)

    # update save directory in config_dict for untrusted models
    config_dict["save_dir"] = untrusted_models_dir
    # Poison untrusted data if poison_fraction is specified
    if config_dict.get("poison_fraction", None) is not None:
        poisoner = BackdoorPoisoner(
            trigger_fn=BackdoorPoisoner.all_trigger, target_label=metadata["label_mapping"]["normal"]
        )
        logging.info(f"Poisoning {poison_fraction * 100:.2f}% of the training data.")
        X_untrusted, y_untrusted = poisoner.poison(
            X_untrusted, y_untrusted, poison_fraction=poison_fraction, random_state=42
        )

    # Combine trusted and untrusted data for training
    X_combined = pd.concat([X_trusted, X_untrusted], ignore_index=True)
    y_combined = pd.concat([y_trusted, y_untrusted], ignore_index=True)

    # Prepare combined data for training
    X_combined_norm, y_combined, X_test_norm, scaler = prepare_data_for_training(
        X_combined, y_combined, X_test, config_dict
    )
    # poison test data
    X_poisoned_test, y_poisoned_test = poisoner.poison(X_test, y_test, poison_fraction=1.0, random_state=42)
    X_poisoned_test, _ = GenericPreProcessor.normalize_data(
        X_poisoned_test, scale_numeric=True, existing_scaler=scaler, columns=norm_cols
    )
    # Train models on combined data
    logging.info("Training models on combined data (trusted + untrusted).")
    train_models(X_combined_norm, y_combined, X_test_norm, y_test, config_dict=config_dict)

    # Load trained models
    trusted_models = load_models(list(config_dict["models_to_train_config"]), trusted_models_dir.as_posix())
    untrusted_models = load_models(list(config_dict["models_to_train_config"]), untrusted_models_dir.as_posix())

    models = {"trusted": trusted_models, "untrusted": untrusted_models}
    data = {
        "X_trusted": X_trusted_norm,
        "y_trusted": y_trusted,
        "X_combined": X_combined_norm,
        "y_combined": y_combined,
        "X_test": X_test_norm,
        "y_test": y_test,
        "poisoned_indices": poisoner.poison_indices_ if hasattr(poisoner, "poison_indices_") else None,
    }

    # reset save directory in config_dict
    config_dict["save_dir"] = save_dir

    return models, data


# --- Example usage on dummy data ---
if __name__ == "__main__":
    set_seed(42)
    DEBUG = True
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

    config = load_yaml("configs/delta_shap_config.yml")

    smote_data_dir = Path(config["smote_data_dir"])
    models_to_train_config = config["models_to_train_config"]
    logging.info(f"Models to train: {models_to_train_config}")
    models_to_train = list(models_to_train_config)
    poison_fraction = config.get("poison_fraction", None)
    train_clean = config.get("train_clean", True)
    use_categorical_embedding = config.get("use_categorical_embedding", True)

    X_train, y_train, X_test, y_test, metadata = load_data(smote_data_dir)

    feature_names = metadata["feature_names"]
    cat_cols = metadata["cat_cols"]
    num_cols = metadata["num_cols"]
    norm_cols = num_cols

    X_train[cat_cols] = X_train[cat_cols].round().astype("int8")
    X_test[cat_cols] = X_test[cat_cols].round().astype("int8")

    if not use_categorical_embedding:
        cat_cols = []
        num_cols = num_cols + cat_cols
        norm_cols = num_cols
        # one-hot encode categorical columns
        X_train = pd.get_dummies(X_train, columns=cat_cols, dtype="int8")
        X_test = pd.get_dummies(X_test, columns=cat_cols, dtype="int8")

    config["cat_cols"] = cat_cols
    config["num_cols"] = num_cols
    config["feature_names"] = feature_names

    # Train trusted and untrusted models
    trained_models, training_data = train_trusted_and_untrusted_models(X_train, y_train, X_test, y_test, config)

    for model_type in trained_models["trusted"]:
        trusted_model = trained_models["trusted"][model_type]
        untrusted_model = trained_models["untrusted"][model_type]

        logging.info(f"Running detection for model type: {model_type}")
        X_combined = training_data["X_combined"]
        X_trusted = training_data["X_trusted"]

        n_suspicious = int(config["poison_fraction"] * (len(X_combined) - len(X_trusted)))

        sus_indices = run_detection(trusted_model, untrusted_model, X_combined, N=n_suspicious)
        poisoned_indices = training_data["poisoned_indices"]
        if poisoned_indices is not None:
            np.save(config["save_dir"] / f"poisoned_indices_{model_type}.npy", poisoned_indices)
        np.save(config["save_dir"] / f"suspicious_indices_{model_type}.npy", sus_indices)
        logging.info(f"Detection completed for model type: {model_type}")
