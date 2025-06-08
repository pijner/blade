import logging
import pandas as pd

from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump
from src.model_factory import MODEL_FACTORY, set_seed


def balance_class_data(X: pd.DataFrame, y, method="SMOTETomek", reduce_majority=True):
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
    # Step 1: Optional reduction of the largest class
    y_counts = y.value_counts()
    if reduce_majority and len(y_counts) > 1:
        label_cap = y_counts.sort_values(ascending=False).iloc[3]

        df = X.copy()
        df["_label"] = y.values  # Avoid index misalignment

        df_balanced = (
            df.groupby("_label")
            .apply(lambda x: x.sample(min(len(x), label_cap), random_state=42))
            .reset_index(drop=True)
        )

        y = df_balanced.pop("_label")
        X = df_balanced

    # Step 2: Apply balancing method
    if method == "SMOTETomek":
        from imblearn.combine import SMOTETomek

        smote_tomek = SMOTETomek(random_state=42)
        X_balanced, y_balanced = smote_tomek.fit_resample(X, y)
    elif method == "SMOTE":
        from imblearn.over_sampling import SMOTE

        smote = SMOTE(random_state=42, k_neighbors=3)
        X_balanced, y_balanced = smote.fit_resample(X, y)
    elif method == "undersample":
        min_count = y.value_counts().min()
        X_balanced = pd.concat([X[y == cls].sample(min_count, random_state=42) for cls in y.unique()])
        y_balanced = pd.concat([y[y == cls].sample(min_count, random_state=42) for cls in y.unique()])
    elif method == "none":
        return X, y  # No balancing applied
    else:
        raise ValueError("Invalid method. Choose from 'SMOTE', 'SMOTETomek', or 'undersample'.")

    return X_balanced, y_balanced


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    models: list[str] = ["logreg", "rf", "mlp"],
    save_models: bool = True,
    model_dir: str = "models",
    cat_cols: list[str] = None,
    num_cols: list[str] = None,
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
    X_train_cont = X_train[num_cols]
    X_train_cat = X_train[cat_cols].astype("int32")

    X_test_cont = X_test[num_cols]
    X_test_cat = X_test[cat_cols].astype("int32")

    cat_dims = [int(X_train[col].max() + 1) for col in cat_cols]
    embed_dims = [min(50, (dim + 1) // 2) for dim in cat_dims]

    num_cont_features = len(num_cols)

    Path(model_dir).mkdir(parents=True, exist_ok=True)

    input_dim = X_train.shape[1]
    num_classes = y_train.max() - y_train.min() + 1

    for model_name in models:
        if model_name not in MODEL_FACTORY:
            raise ValueError(f"Model '{model_name}' is not supported. Choose from {list(MODEL_FACTORY.keys())}.")

    models = {
        name: MODEL_FACTORY[name](num_cont_features, cat_dims, embed_dims, num_classes)
        if name in ["mlp"]
        else MODEL_FACTORY[name]
        for name in models
        if name in MODEL_FACTORY
    }

    results = {}

    for name, model in models.items():
        print(f"Training: {name.upper()}")
        if name == "mlp":
            model.fit(X_train_cont, X_train_cat, y_train, epochs=30, batch_size=1024, lr=1e-4)
            preds = model.predict(X_test_cont, X_test_cat)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results[name] = acc

        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, preds, digits=3))

        if name == "xgb":
            # plot feature importances for XGBoost
            import matplotlib.pyplot as plt
            import xgboost as xgb

            xgb.plot_importance(model)
            plt.title("Feature Importances")
            plt.tight_layout()
            plt.savefig(f"{model_dir}/{name}_feature_importances.png")

            # print feature importances
            feature_importances = pd.Series(model.feature_importances_, index=X_train.columns)
            print("\nFeature Importances:")
            print(feature_importances.sort_values(ascending=False))

        if save_models:
            path = f"{model_dir}/{name}_clf.joblib"
            dump(model, path)
            print(f"Saved model to {path}")

    return results


if __name__ == "__main__":
    from src.pre_processing import ToNIoTPreProcessor

    set_seed(42)
    DEBUG = True
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    preprocessor = ToNIoTPreProcessor(
        "data/ton_iot/Processed_datasets/Processed_Network_dataset",
        "data/ton_iot/Train_Test_datasets/Train_Test_Network_dataset/train_test_network.csv",
        save_processed=True,
        save_path="data/ton_iot/preprocessed_data_splits",
    )
    X_train, X_test, y_train, y_test, feature_names, cat_cols, num_cols = preprocessor.get_ton_iot_network_data(
        label_col="type",
        scale_numeric=True,
        check_duplicates=True,
        try_load_preprocessed=True,
        drop_labels=["mitm", "ransomware"],
    )

    logging.info("Training and testing data loaded successfully.")

    print("\nClass value counts (before balancing):")
    print(y_train.value_counts(normalize=True))
    print(y_test.value_counts(normalize=True))
    print("-" * 50)

    X_train, y_train = balance_class_data(X_train, y_train, method="undersample", reduce_majority=True)

    print("\nClass value counts (after balancing):")
    print(y_train.value_counts(normalize=True))
    print(y_test.value_counts(normalize=True))
    print("-" * 50)

    # Train models and evaluate
    results = train_models(X_train, y_train, X_test, y_test, models=["xgb", "rf", "mlp", "logreg"])
    print("Training results:", results)
