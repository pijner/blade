import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE, SMOTENC


def balance_class_data(X: pd.DataFrame, y, method="SMOTENC", reduce_majority=True, cat_cols=None):
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
        smote_tomek = SMOTETomek(random_state=42)
        X_balanced, y_balanced = smote_tomek.fit_resample(X, y)
    elif method == "SMOTE":
        smote = SMOTE(random_state=42, k_neighbors=3)
        X_balanced, y_balanced = smote.fit_resample(X, y)
    elif method == "SMOTENC":
        assert cat_cols is not None, "cat_cols must be provided for SMOTENC"

        smote = SMOTENC(categorical_features=cat_cols, random_state=42, k_neighbors=1)
        X_balanced, y_balanced = smote.fit_resample(X, y)
    elif method == "undersample":
        min_count = y.value_counts().min()
        X_balanced = pd.concat([X[y == cls].sample(min_count, random_state=42) for cls in y.unique()])
        y_balanced = pd.concat([y[y == cls].sample(min_count, random_state=42) for cls in y.unique()])
    elif method == "none":
        return X, y  # No balancing applied
    else:
        raise ValueError("Invalid method. Choose from 'SMOTE', 'SMOTETomek', 'SMOTENC', or 'undersample'.")

    return X_balanced, y_balanced
