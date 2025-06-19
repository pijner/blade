import logging
import json
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump
from src.model_factory import MODEL_FACTORY, set_seed


def balance_class_data(X: pd.DataFrame, y, method="SMOTETomek", reduce_majority=True, cat_cols=None):
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
        label_cap = y_counts.sort_values(ascending=False).iloc[2]

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
    elif method == "SMOTENC":
        assert cat_cols is not None, "cat_cols must be provided for SMOTENC"
        from imblearn.over_sampling import SMOTENC

        smote = SMOTENC(categorical_features=cat_cols, random_state=42, k_neighbors=3)
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
    poisoned_test_data: pd.DataFrame = None,
    poisoned_test_labels: pd.Series = None,
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

    if poisoned_test_data is not None and poisoned_test_labels is not None:
        X_poisoned_test_cont = poisoned_test_data[num_cols]
        X_poisoned_test_cat = poisoned_test_data[cat_cols].astype("int32")
        y_poisoned_test = poisoned_test_labels

    cat_dims = [int(X_train[col].max() + 1) for col in cat_cols]
    embed_dims = [min(50, int(round(dim**0.25))) for dim in cat_dims]

    num_cont_features = len(num_cols)

    Path(model_dir).mkdir(parents=True, exist_ok=True)

    num_classes = y_train.max() - y_train.min() + 1

    for model_name in models:
        if model_name not in MODEL_FACTORY:
            raise ValueError(f"Model '{model_name}' is not supported. Choose from {list(MODEL_FACTORY.keys())}.")

    models = {
        name: MODEL_FACTORY[name](
            num_cont_features, cat_dims, embed_dims, num_classes, X_train.columns.tolist(), cat_cols
        )
        if name in ["mlp", "tf_mlp"]
        else MODEL_FACTORY[name]
        for name in models
        if name in MODEL_FACTORY
    }

    results = {}

    for name, model in models.items():
        print(f"Training: {name.upper()}")
        if name == "tf_mlp":
            model.fit(X_train_cont, X_train_cat, y_train, epochs=15, batch_size=1024 * 10, lr=1e-3)
            preds = model.predict(X_test_cont, X_test_cat)
            if poisoned_test_data is not None and poisoned_test_labels is not None:
                poisoned_preds = model.predict(X_poisoned_test_cont, X_poisoned_test_cat)
                poisoned_acc = accuracy_score(y_poisoned_test, poisoned_preds)
                print(f"Poisoned Test Accuracy: {poisoned_acc:.4f}")
        elif name in ["mlp"]:
            model.fit(X_train_cont, X_train_cat, y_train, epochs=30, batch_size=1024, lr=1e-4)
            preds = model.predict(X_test_cont, X_test_cat)
            if poisoned_test_data is not None and poisoned_test_labels is not None:
                poisoned_preds = model.predict(X_poisoned_test_cont, X_poisoned_test_cat)
                poisoned_acc = accuracy_score(y_poisoned_test, poisoned_preds)
                print(f"Poisoned Test Accuracy: {poisoned_acc:.4f}")
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            if poisoned_test_data is not None and poisoned_test_labels is not None:
                poisoned_preds = model.predict(poisoned_test_data)
                poisoned_acc = accuracy_score(poisoned_test_labels, poisoned_preds)
                print(f"Poisoned Test Accuracy: {poisoned_acc:.4f}")
        acc = accuracy_score(y_test, preds)
        results[name] = acc

        # dump accuracy and classification report to a file
        with open(f"{model_dir}/{name}_results.txt", "w") as f:
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(classification_report(y_test, preds, digits=3))
            if poisoned_test_data is not None and poisoned_test_labels is not None:
                f.write(f"Poisoned Test Accuracy: {poisoned_acc:.4f}\n")
                f.write(classification_report(y_poisoned_test, poisoned_preds, digits=3))

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

        # show confusion matrix
        from sklearn.metrics import ConfusionMatrixDisplay
        import matplotlib.pyplot as plt

        disp = ConfusionMatrixDisplay.from_predictions(
            y_test,
            preds,
            normalize="true",
        )
        disp.ax_.set_title(f"Confusion Matrix for {name.upper()}")
        plt.savefig(f"{model_dir}/{name}_confusion_matrix.png")

        if poisoned_test_data is not None and poisoned_test_labels is not None:
            disp_poisoned = ConfusionMatrixDisplay.from_predictions(
                y_poisoned_test,
                poisoned_preds,
                normalize="true",
            )
            disp_poisoned.ax_.set_title(f"Poisoned Test Confusion Matrix for {name.upper()}")
            plt.savefig(f"{model_dir}/{name}_poisoned_confusion_matrix.png")

    return results


def normalize_data(df: pd.DataFrame, scale_numeric: bool = True, existing_scaler=None, columns=None):
    """
    Normalize numeric features in the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with numeric features to normalize.
    scale_numeric : bool, default=True
        Whether to apply StandardScaler to numeric features.

    Returns
    -------
    pandas.DataFrame
        DataFrame with normalized numeric features.
    """
    from sklearn.preprocessing import StandardScaler

    if scale_numeric:
        df = df.copy()  # Avoid modifying the original DataFrame
        scaler = StandardScaler() if existing_scaler is None else existing_scaler
        columns = columns or df.columns.tolist()
        df[columns] = scaler.fit_transform(df[columns]) if existing_scaler is None else scaler.transform(df[columns])
        logging.info("Numeric features scaled using StandardScaler.")
    return df, scaler


def group_and_relabel_classes(
    X: pd.DataFrame,
    y: pd.Series,
    combine_labels: list[int],
    categorical_label_mapping: dict,
    new_label: int = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Groups the specified combine_labels into a single class and relabels all labels to be contiguous.

    Parameters
    ----------
    X : pd.DataFrame
        Feature data.
    y : pd.Series
        Label data.
    combine_labels : List[int]
        List of labels to be combined into a single new class.
    new_label : int, optional
        The new label to assign to the combined group. If None, uses the minimum label in combine_labels.

    Returns
    -------
    X_out : pd.DataFrame
        Unchanged feature data.
    y_out : pd.Series
        Relabeled target values.
    """
    if new_label is None:
        new_label = min(combine_labels)

    # Map all combine_labels to the new_label
    y_mapped = y.copy()
    y_mapped[y.isin(combine_labels)] = new_label

    # Make labels contiguous
    unique_labels = sorted(y_mapped.unique())
    label_mapping = {old: new for new, old in enumerate(unique_labels)}
    y_reindexed = y_mapped.map(label_mapping)

    if categorical_label_mapping is not None:
        categorical_label_mapping = {k: label_mapping[v] for k, v in categorical_label_mapping.items()}

    return X.copy(), y_reindexed, categorical_label_mapping


if __name__ == "__main__":
    set_seed(42)
    DEBUG = True
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

    smote_data_dir = Path("data/ton_iot/SMOTE_train_data")
    smote_train_data = smote_data_dir / "smote_train_data.csv"
    smote_train_labels = smote_data_dir / "smote_train_labels.csv"
    test_data = smote_data_dir / "test_data.csv"
    test_labels = smote_data_dir / "test_labels.csv"

    if not smote_train_data.exists() or not smote_train_labels.exists():
        from src.pre_processing import ToNIoTPreProcessor

        logging.info("SMOTE training data not found. Creating directory for SMOTE training data.")
        smote_data_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created directory for SMOTE training data: {smote_data_dir}")

        preprocessor = ToNIoTPreProcessor(
            "data/ton_iot/Processed_datasets/Processed_Network_dataset",
            "data/ton_iot/Train_Test_datasets/Train_Test_Network_dataset/train_test_network.csv",
            save_processed=True,
            save_path="data/ton_iot/preprocessed_data_splits",
        )
        X_train, X_test, y_train, y_test, feature_names, cat_cols, num_cols = preprocessor.get_ton_iot_network_data(
            label_col="type",
            scale_numeric=False,
            check_duplicates=True,
            try_load_preprocessed=True,
            drop_labels=["mitm", "ransomware"],
            group_labels=["dos", "ddos"],
        )

        print("\nClass value counts (before balancing):")
        print(y_train.value_counts(normalize=True))
        print(y_test.value_counts(normalize=True))
        print("-" * 50)
        X_train, y_train = balance_class_data(X_train, y_train, method="SMOTE", reduce_majority=True, cat_cols=cat_cols)
        print("\nClass value counts (after balancing):")
        print(y_train.value_counts(normalize=True))
        print(y_test.value_counts(normalize=True))
        print("-" * 50)

        # Save the balanced training data
        X_train.to_csv(smote_train_data, index=False)
        y_train.to_csv(smote_train_labels, index=False)
        X_test.to_csv(test_data, index=False)
        y_test.to_csv(test_labels, index=False)

        # copy metadata json file from preprocessor
        with open(smote_data_dir.joinpath("metadata.json"), "w") as f:
            json.dump(preprocessor.metadata, f, indent=2)

    else:
        logging.info("SMOTE training data found. Loading existing data.")
        X_train = pd.read_csv(smote_train_data).astype("float32")
        y_train = pd.read_csv(smote_train_labels).squeeze()
        X_test = pd.read_csv(test_data).astype("float32")
        y_test = pd.read_csv(test_labels).squeeze().astype("uint8")

    # load metadata from json file

    with open(smote_data_dir.joinpath("metadata.json"), "r") as f:
        metadata = json.load(f)
    feature_names = metadata["feature_names"]
    cat_cols = metadata["cat_cols"]
    num_cols = metadata["num_cols"]
    norm_cols = num_cols

    use_categorical_embedding = True
    if not use_categorical_embedding:
        cat_cols = []
        num_cols = num_cols + cat_cols
        norm_cols = num_cols
    else:
        X_train[cat_cols] = X_train[cat_cols].round().astype("int32")

    # nornalize data
    X_train_norm, scaler = normalize_data(X_train, scale_numeric=True, columns=norm_cols)
    X_test_norm, _ = normalize_data(X_test, scale_numeric=True, existing_scaler=scaler, columns=norm_cols)

    logging.info("Training and testing data loaded successfully.")

    # Train models and evaluate
    results = train_models(
        X_train_norm,
        y_train,
        X_test_norm,
        y_test,
        cat_cols=cat_cols,
        num_cols=num_cols,
        models=["tf_mlp"],
        model_dir="models/clean_models",
    )
    print("Training results:", results)

    # Poison data and retrain
    from src.backdoor_attack import BackdoorPoisoner

    poisoner = BackdoorPoisoner(
        trigger_fn=BackdoorPoisoner.all_trigger,
        target_label=metadata["label_mapping"]["normal"],  # Target label for poisoned samples
    )

    X_poisoned, y_poisoned = poisoner.poison(X_train, y_train, poison_fraction=0.10, random_state=42)
    X_poisoned_norm, poisoned_scaled = normalize_data(X_poisoned, scale_numeric=True, columns=norm_cols)
    X_test_norm, _ = normalize_data(X_test, scale_numeric=True, existing_scaler=poisoned_scaled, columns=norm_cols)

    # poison test data
    X_poisoned_test, y_poisoned_test = poisoner.poison(X_test, y_test, poison_fraction=1.0, random_state=42)

    results = train_models(
        X_poisoned_norm,
        y_train,
        X_test_norm,
        y_test,
        cat_cols=cat_cols,
        num_cols=num_cols,
        models=["xgb"],
        model_dir="models/poisoned_models",
        poisoned_test_data=X_poisoned_test,
        poisoned_test_labels=y_poisoned_test,
    )
    print("Training results (after poisoning):", results)
