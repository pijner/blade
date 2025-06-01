import pandas as pd
import numpy as np
import logging

from typing import Optional
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer


class ToNIoTPreProcessor:
    def __init__(self, processed_data_dir: str, test_data_path: str):
        self.processed_data_dir = Path(processed_data_dir).as_posix()
        self.test_data_path = Path(test_data_path).as_posix()
        self.column_transformer = None

        self.network_default_drop_cols = [
            "ts",
            "src_ip",
            "dst_ip",
            "dns_query",
            "ssl_subject",
            "ssl_issuer",
            "http_uri",
            "http_referrer",
            "http_user_agent",
            "http_orig_mime_types",
            "http_resp_mime_types",
            "weird_addl",
            "type",
        ]

    def _load_data(self, csv_path: str, drop_cols: Optional[list[str]] = None) -> pd.DataFrame:
        """
        Load the ToN-IoT data from the specified csv path.
        """
        try:
            df = pd.read_csv(csv_path)
            logging.info(f"Loaded processed data from {csv_path}")
            if drop_cols is not None:
                df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")

            return df
        except FileNotFoundError as e:
            logging.error(f"Processed data file not found at {csv_path}")
            raise e

    def preprocess_network_data(
        self,
        df: pd.DataFrame,
        label_col: str = "label",
        scale_numeric: bool = True,
        use_ordinal_encoding: bool = True,
    ):
        """
        Preprocess the ToN-IoT Network dataset for binary classification.

        Parameters
        ----------
        df : pandas.DataFrame
            Raw dataframe loaded from IoT-network.csv.
        label_col : str, default='label'
            Name of the label column.
        scale_numeric : bool, default=True
            Whether to scale numeric features using StandardScaler.

        Returns
        -------
        X : pandas.DataFrame
            Processed feature matrix.
        y : pandas.Series
            Target labels.
        feature_names : list of str
            Feature names after preprocessing.
        """

        df = df.dropna()

        df["src_bytes"] = pd.to_numeric(df["src_bytes"], errors="coerce")
        df = df.dropna(subset=["src_bytes"])

        y = df[label_col].astype(int)
        X = df.drop(columns=[label_col])

        # Identify column types
        cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
        num_cols = X.select_dtypes(include=["int32", "float32"]).columns.tolist()

        for col in cat_cols:
            types = X[col].map(type).unique()
            if len(types) > 1:
                logging.warning(f"⚠️ Column '{col}' contains mixed types: {types}")

        # Construct transformer
        if self.column_transformer is None:
            if use_ordinal_encoding:
                logging.info("Using Ordinal Encoding for categorical features.")
                cat_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype="np.float32")
            else:
                logging.info("Using One-Hot Encoding for categorical features.")
                cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)

            self.column_transformer = ColumnTransformer(
                [
                    ("num", StandardScaler(copy=False) if scale_numeric else "passthrough", num_cols),
                    ("cat", cat_encoder, cat_cols),
                ]
            )

            X_processed = self.column_transformer.fit_transform(X)
        else:
            X_processed = self.column_transformer.transform(X)

        if hasattr(X_processed, "toarray"):
            X_processed = X_processed.toarray()

        cat_features = self.column_transformer.named_transformers_["cat"].get_feature_names_out(cat_cols)
        feature_names = num_cols + list(cat_features)

        return pd.DataFrame(X_processed, columns=feature_names), y.reset_index(drop=True), feature_names

    def get_ton_iot_network_data(
        self,
        label_col: str,
        drop_cols: Optional[list[str]] = None,
        scale_numeric: bool = True,
        check_duplicates: bool = True,
    ):
        """
        Load and preprocess the ToN-IoT Network dataset using official train/test split.

        Returns
        -------
        X_train, X_test : pandas.DataFrame
        y_train, y_test : pandas.Series
        feature_names : list of str
        """
        # load all csv files from the processed data directory
        csv_files = list(Path(self.processed_data_dir).glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No processed data files found in {self.processed_data_dir}")

        all_dfs = []
        for csv_file in csv_files:
            logging.info(f"Loading data from data file: {csv_file.name}")
            df = self._load_data(csv_file, drop_cols=drop_cols or self.network_default_drop_cols)

            # convert 64 bit values to 32 bit
            float_cols = df.select_dtypes(include="float64").columns
            int_cols = df.select_dtypes(include="int64").columns
            df[float_cols] = df[float_cols].astype("float32")
            df[int_cols] = df[int_cols].astype("int32")

            all_dfs.append(df)

        df_full = pd.concat(all_dfs, ignore_index=True)
        df_test = self._load_data(self.test_data_path, drop_cols=drop_cols or self.network_default_drop_cols)

        # Align schema
        df_full = df_full[df_test.columns]
        logging.info(f"Full dataset shape: {df_full.shape}, Test dataset shape: {df_test.shape}")

        # Preprocess both
        df_full = self.preprocess_network_data(df_full, label_col=label_col, scale_numeric=scale_numeric)
        df_test = self.preprocess_network_data(df_test, label_col=label_col, scale_numeric=scale_numeric)

        # Unpack outputs
        X_full, y_full, _ = df_full
        X_test, y_test, _ = df_test

        # Join full for label conflict check
        df_full_combined = X_full.copy()
        df_full_combined[label_col] = y_full

        conflicts, conflict_count = self.check_label_conflicts(df_full_combined, label_col=label_col)
        if conflict_count > 0:
            logging.warning(f"⚠️ Found {conflict_count} label conflicts in training data.\n{conflicts.head()}")
        else:
            logging.info("✅ No label conflicts found.")

        if check_duplicates:
            logging.info("Duplicate Check:")
            logging.info(f"  Full dataset: {X_full.duplicated().sum()} duplicate rows")
            logging.info(f"  Test dataset: {X_test.duplicated().sum()} duplicate rows")

        # Create training set by subtracting test from full
        df_train = pd.concat([X_full.assign(_label=y_full), X_test.assign(_label=y_test), X_test.assign(_label=y_test)])
        df_train = df_train.drop_duplicates(keep=False)

        y_train = df_train["_label"]
        X_train = df_train.drop(columns=["_label"])

        float_cols = X_train.select_dtypes(include="float64").columns
        int_cols = X_train.select_dtypes(include="int64").columns

        X_train[float_cols] = X_train[float_cols].astype("float32")
        X_test[float_cols] = X_test[float_cols].astype("float32")
        X_train[int_cols] = X_train[int_cols].astype("int32")
        X_test[int_cols] = X_test[int_cols].astype("int32")

        y_train = y_train.astype(np.int32)
        y_test = y_test.astype(np.int32)

        logging.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        logging.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

        return X_train, X_test, y_train, y_test, X_train.columns.tolist()

    def check_label_conflicts(self, df, label_col="label"):
        """
        Check for label conflicts in the dataset: rows with identical feature values
        but different labels.

        Parameters
        ----------
        df : pandas.DataFrame
            Preprocessed dataframe including features and a label column.
        label_col : str, default='label'
            Name of the label column to check for conflicts.

        Returns
        -------
        conflicts : pandas.DataFrame
            DataFrame of conflicting feature rows with multiple labels.
        conflict_count : int
            Total number of unique conflicting feature groups.
        """
        # Drop the label column to group by features
        feature_cols = [col for col in df.columns if col != label_col]

        # Group by features and count unique labels in each group
        label_variation = df.groupby(feature_cols)[label_col].nunique()

        # Conflicts occur where more than 1 label exists for the same features
        conflicts = label_variation[label_variation > 1]

        # Convert back to dataframe format for inspection
        conflict_rows = conflicts.reset_index().merge(df, on=feature_cols, how="left")

        return conflict_rows, len(conflicts)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    preprocessor = ToNIoTPreProcessor(
        "data/ton_iot/Processed_datasets/Processed_Network_dataset",
        "data/ton_iot/Train_Test_datasets/Train_Test_Network_dataset/train_test_network.csv",
    )
    X_train, X_test, y_train, y_test, feature_names = preprocessor.get_ton_iot_network_data(
        label_col="label",
        scale_numeric=True,
        check_duplicates=True,
    )

    print(f"Features: {feature_names}")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
