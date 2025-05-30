import logging
import pandas as pd

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump

MODEL_FACTORY = {
    "logreg": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "rf": RandomForestClassifier(n_estimators=100, class_weight="balanced"),
    "svm": SVC(probability=True, class_weight="balanced"),
    "mlp": MLPClassifier(hidden_layer_sizes=(32, 32), max_iter=300),
    "xgb": XGBClassifier(scale_pos_weight=1.0),
}


def balance_class_data(X, y):
    """
    Balance the dataset by undersampling the majority class.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix.
    y : pandas.Series
        Labels.

    Returns
    -------
    X_balanced, y_balanced : pandas.DataFrame, pandas.Series
        Balanced feature matrix and labels.
    """
    class_counts = y.value_counts()
    min_count = class_counts.min()

    balanced_X = pd.concat([X[y == cls].sample(min_count, random_state=42) for cls in class_counts.index])

    balanced_y = pd.concat([y[y == cls].sample(min_count, random_state=42) for cls in class_counts.index])

    return balanced_X, balanced_y


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    models: list[str] = ["logreg", "rf", "mlp"],
    save_models: bool = True,
    model_dir: str = "models",
):
    """
    Train and evaluate multiple classifiers on the given tabular dataset.

    Parameters
    ----------
    X_train, X_test : pandas.DataFrame
        Feature matrices.
    y_train, y_test : pandas.Series
        Labels.
    save_models : bool, default=True
        Whether to save trained models using joblib.
    model_dir : str, default='models'
        Directory path to save model files.

    Returns
    -------
    results : dict
        Dictionary of model names to their accuracy scores.
    """
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    for model_name in models:
        if model_name not in MODEL_FACTORY:
            raise ValueError(f"Model '{model_name}' is not supported. Choose from {list(MODEL_FACTORY.keys())}.")

    models = {name: MODEL_FACTORY[name] for name in models if name in MODEL_FACTORY}

    results = {}

    for name, model in models.items():
        print(f"Training: {name.upper()}")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results[name] = acc

        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, preds, digits=3))

        if save_models:
            path = f"{model_dir}/{name}_clf.joblib"
            dump(model, path)
            print(f"Saved model to {path}")

    return results


if __name__ == "__main__":
    from src.pre_processing import ToNIoTPreProcessor

    DEBUG = True
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    preprocessor = ToNIoTPreProcessor(
        "data/ton_iot/Processed_datasets/Processed_Network_dataset/Network_dataset_1.csv",
        "data/ton_iot/Train_Test_datasets/Train_Test_Network_dataset/train_test_network.csv",
    )
    X_train, X_test, y_train, y_test, feature_names = preprocessor.get_ton_iot_network_data(
        label_col="label",
        scale_numeric=True,
        check_duplicates=True,
    )
    X_train, y_train = balance_class_data(X_train, y_train)

    if DEBUG:
        import seaborn as sns
        import matplotlib.pyplot as plt

        xgb = MODEL_FACTORY["xgb"]
        xgb.fit(X_train, y_train)
        print("XGBoost model trained.")

        importances = xgb.feature_importances_
        top_features = pd.Series(importances, index=X_train.columns).sort_values(ascending=False)
        top_features.head(20).plot(kind="barh", title="Top 20 XGBoost Feature Importances")
        plt.tight_layout()
        plt.show()

        for col in ["duration", "dst_port", "proto"]:
            sns.kdeplot(data=X_train.assign(label=y_train), x=col, hue="label")
            plt.title(f"Distribution of {col} by label")
            plt.show()

        print("\nClass value counts:")
        print(y_train.value_counts(normalize=True))
        print(y_test.value_counts(normalize=True))
        print("-" * 50)

    # Train models and evaluate
    results = train_models(X_train, y_train, X_test, y_test, models=["logreg", "rf", "mlp"])
    print("Training results:", results)
