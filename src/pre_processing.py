import pandas as pd
import numpy as np
import logging

from typing import Optional
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer


class ToNIoTPreProcessor:
    def __init__(
        self,
        processed_data_dir: str,
        test_data_path: str,
        save_processed: bool = False,
        save_path: Optional[str] = None,
    ):
        self.processed_data_dir = Path(processed_data_dir).as_posix()
        self.test_data_path = Path(test_data_path).as_posix()
        self.save_processed = save_processed
        if save_processed:
            if save_path is None:
                raise ValueError("save_path must be provided if save_processed is True")

            Path(save_path).mkdir(parents=True, exist_ok=True)

        self.save_path = Path(save_path).as_posix() if save_path else None

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

            # convert 64 bit values to 32 bit
            float_cols = df.select_dtypes(include="float64").columns
            int_cols = df.select_dtypes(include="int64").columns
            df[float_cols] = df[float_cols].astype("float32")
            df[int_cols] = df[int_cols].astype("int32")

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

        y = df[label_col].astype(np.int32)
        X = df.drop(columns=[label_col])

        # Identify column types
        cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
        num_cols = X.select_dtypes(include=["int32", "float32", "int64", "float64"]).columns.tolist()

        for col in cat_cols:
            types = X[col].map(type).unique()
            if len(types) > 1:
                logging.warning(f"⚠️ Column '{col}' contains mixed types: {types}")

        # Construct transformer
        if self.column_transformer is None:
            if use_ordinal_encoding:
                logging.info("Using Ordinal Encoding for categorical features.")
                cat_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float32)
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
        try_load_preprocessed: bool = True,
    ):
        """
        Load and preprocess the ToN-IoT Network dataset using official train/test split.

        Returns
        -------
        X_train, X_test : pandas.DataFrame
        y_train, y_test : pandas.Series
        feature_names : list of str
        """
        if try_load_preprocessed:
            try:
                logging.info("Attempting to load preprocessed data...")
                return self.load_preprocessed_data(self.save_path)
            except FileNotFoundError:
                logging.warning("Preprocessed data not found, proceeding with raw data loading.")

        # load all csv files from the processed data directory
        csv_files = list(Path(self.processed_data_dir).glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No processed data files found in {self.processed_data_dir}")

        all_dfs = []
        for csv_file in csv_files:
            logging.info(f"Loading data from data file: {csv_file.name}")
            df = self._load_data(csv_file, drop_cols=drop_cols or self.network_default_drop_cols)

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
            logging.info("Resolving conflicts by removing all conflicting rows...")
            df_full_combined = self.resolve_conflicts(df_full_combined, conflicts, label_col=label_col)
            X_full = df_full_combined.drop(columns=[label_col])
            y_full = df_full_combined[label_col]
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

        if self.save_processed:
            X_train.to_csv(Path(self.save_path).joinpath("X_train.csv"), index=False)
            X_test.to_csv(Path(self.save_path).joinpath("X_test.csv"), index=False)
            y_train.to_csv(Path(self.save_path).joinpath("y_train.csv"), index=False)
            y_test.to_csv(Path(self.save_path).joinpath("y_test.csv"), index=False)
            logging.info(f"Processed data saved to {self.save_path}")

        return X_train, X_test, y_train, y_test, X_train.columns.tolist()

    def load_preprocessed_data(self, data_path: str):
        """
        Load preprocessed data from the specified path.

        Parameters
        ----------
        data_path : str
            Path to the preprocessed data directory.

        Returns
        -------
        X_train, X_test : pandas.DataFrame
            Feature matrices for training and testing.
        y_train, y_test : pandas.Series
            Target labels for training and testing.
        feature_names : list of str
            Names of the features.
        """
        X_train = pd.read_csv(Path(data_path).joinpath("X_train.csv"))
        X_test = pd.read_csv(Path(data_path).joinpath("X_test.csv"))
        y_train = pd.read_csv(Path(data_path).joinpath("y_train.csv")).squeeze()
        y_test = pd.read_csv(Path(data_path).joinpath("y_test.csv")).squeeze()

        return X_train, X_test, y_train, y_test

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

    def resolve_conflicts(self, df, conflict_rows, label_col="label"):
        """
        Remove all rows from the original dataframe that are part of a label conflict.

        Parameters
        ----------
        df : pandas.DataFrame
            Original dataframe including features and label column.
        conflict_rows : pandas.DataFrame
            DataFrame of rows involved in conflicts, as returned by check_label_conflicts().
        label_col : str, default='label'
            Name of the label column.

        Returns
        -------
        df_clean : pandas.DataFrame
            DataFrame with all conflicting rows removed.
        """
        feature_cols = [col for col in df.columns if col != label_col]

        # Create a tuple for each row of features to identify conflicts
        conflict_feature_tuples = set(tuple(row) for row in conflict_rows[feature_cols].to_numpy())

        # Filter out any row in df that has features matching a conflict
        df_clean = df[~df[feature_cols].apply(tuple, axis=1).isin(conflict_feature_tuples)].copy()

        return df_clean


if __name__ == "__main__":
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

    print(f"Features: {feature_names}")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
