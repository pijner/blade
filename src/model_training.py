import logging
import pandas as pd

from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump
from src.model_factory import MODEL_FACTORY, set_seed


def balance_class_data(X, y, method="SMOTETomek"):
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
    if method == "SMOTETomek":
        from imblearn.combine import SMOTETomek

        smote_tomek = SMOTETomek(random_state=42)
        X_balanced, y_balanced = smote_tomek.fit_resample(X, y)
        return X_balanced, y_balanced
    elif method == "SMOTE":
        from imblearn.over_sampling import SMOTE

        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        return X_balanced, y_balanced

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

    input_dim = X_train.shape[1]
    num_classes = len(y_train.unique())

    for model_name in models:
        if model_name not in MODEL_FACTORY:
            raise ValueError(f"Model '{model_name}' is not supported. Choose from {list(MODEL_FACTORY.keys())}.")

    models = {
        name: MODEL_FACTORY[name](input_dim, num_classes) if name in ["mlp", "tabnet"] else MODEL_FACTORY[name]
        for name in models
        if name in MODEL_FACTORY
    }

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

    set_seed(42)
    DEBUG = True
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    preprocessor = ToNIoTPreProcessor(
        "data/ton_iot/Processed_datasets/Processed_Network_dataset",
        "data/ton_iot/Train_Test_datasets/Train_Test_Network_dataset/train_test_network.csv",
        save_processed=True,
        save_path="data/ton_iot/preprocessed_data_splits",
    )
    X_train, X_test, y_train, y_test, feature_names = preprocessor.get_ton_iot_network_data(
        label_col="label", scale_numeric=True, check_duplicates=True, try_load_preprocessed=True
    )

    logging.info("Training and testing data loaded successfully.")

    X_train, y_train = balance_class_data(X_train, y_train, method="SMOTEx")

    print("\nClass value counts:")
    print(y_train.value_counts(normalize=True))
    print(y_test.value_counts(normalize=True))
    print("-" * 50)

    # Train models and evaluate
    results = train_models(X_train, y_train, X_test, y_test, models=["mlp", "xgb"])
    print("Training results:", results)
