import pandas as pd
import numpy as np
import logging
import json

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
        self.label_encoder = None

        self.cat_cols = None
        self.num_cols = None
        self.metadata = {}

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
            "label",
            "http_response_body_len",
            "http_status_code",
            "ssl_version",
            "ssl_established",
            "ssl_cipher",
            "http_version",
            "weird_notice",
            "http_request_body_len",
            "ssl_resumed",
            "missed_bytes",
            "http_trans_depth",
            "weird_name",
            "http_method",
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

            return self.convert_to_32bit(df)
        except FileNotFoundError as e:
            logging.error(f"Processed data file not found at {csv_path}")
            raise e

    def convert_to_32bit(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert all float64 and int64 columns in the DataFrame to float32 and int32 respectively.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame to convert.

        Returns
        -------
        pandas.DataFrame
            The DataFrame with converted column types.
        """
        float_cols = df.select_dtypes(include="float64").columns
        int_cols = df.select_dtypes(include="int64").columns

        df[float_cols] = df[float_cols].astype("float32")
        df[int_cols] = df[int_cols].astype("int32")

        return df

    def preprocess_network_data(
        self,
        df: pd.DataFrame,
        label_col: str = "type",
        scale_numeric: bool = True,
        use_ordinal_encoding: bool = True,
    ):
        """
        Preprocess the ToN-IoT Network dataset for binary classification.

        Parameters
        ----------
        df : pandas.DataFrame
            Raw dataframe loaded from IoT-network.csv.
        label_col : str, default='type'
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

        if label_col == "type":
            # encode categorical labels in 'type'
            logging.info("Encoding 'type' column as categorical labels.")
            if self.label_encoder is None:
                self.label_encoder = LabelEncoder()
                df[label_col] = self.label_encoder.fit_transform(df[label_col])
                # print label mapping
                logging.info("Label mapping")
                # write label mapping to file
                with open(Path(self.save_path).joinpath("label_mapping.txt"), "w") as f:
                    for label, encoded in zip(
                        self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)
                    ):
                        self.metadata["label_mapping"] = {
                            **self.metadata.get("label_mapping", {}),
                            **{label: int(encoded)},
                        }
                        f.write(f"{label}: {encoded}\n")
                        logging.info(f"{label}: {encoded}")
            else:
                df[label_col] = self.label_encoder.transform(df[label_col])

        y = df[label_col].astype(np.int32)
        X = df.drop(columns=[label_col])

        # Identify column types
        if self.cat_cols is None:
            self.cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
        if self.num_cols is None:
            self.num_cols = X.select_dtypes(include=["int32", "float32", "int64", "float64"]).columns.tolist()

        for col in self.cat_cols:
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
                cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.uint8)

            self.column_transformer = ColumnTransformer(
                [
                    ("num", StandardScaler(copy=False) if scale_numeric else "passthrough", self.num_cols),
                    ("cat", cat_encoder, self.cat_cols),
                ]
            )

            X_processed = self.column_transformer.fit_transform(X)
        else:
            X_processed = self.column_transformer.transform(X)

        if hasattr(X_processed, "toarray"):
            X_processed = X_processed.toarray()

        cat_features = self.column_transformer.named_transformers_["cat"].get_feature_names_out(self.cat_cols)
        feature_names = self.num_cols + list(cat_features)

        X_processed = self.convert_to_32bit(pd.DataFrame(X_processed, columns=feature_names))

        return X_processed, y.reset_index(drop=True), feature_names, self.cat_cols, self.num_cols

    def get_ton_iot_network_data(
        self,
        label_col: str,
        drop_cols: Optional[list[str]] = None,
        drop_labels: Optional[list[str]] = None,
        group_labels: Optional[list[str]] = None,
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

        print("-" * 50)
        logging.info(f"Label distribution in full dataset:\n{df_full[label_col].value_counts()}")
        logging.info(f"Label distribution in test dataset:\n{df_test[label_col].value_counts()}")
        print("-" * 50)

        # Align schema
        df_full = df_full[df_test.columns]
        logging.info(f"Full dataset shape: {df_full.shape}, Test dataset shape: {df_test.shape}")

        # Drop specified labels if provided
        if drop_labels is not None:
            logging.info(f"Dropping labels: {drop_labels}")
            df_full = df_full[~df_full[label_col].isin(drop_labels)]
            df_test = df_test[~df_test[label_col].isin(drop_labels)]

        # Group labels if specified
        if group_labels is not None:
            logging.info(f"Grouping labels: {group_labels}")
            group_name = "_".join(group_labels)
            for group_label in group_labels:
                df_full.loc[df_full[label_col] == group_label, label_col] = group_name
                df_test.loc[df_test[label_col] == group_label, label_col] = group_name

        # Check for label conflicts before pre-processing
        conflicts, conflict_count = self.check_label_conflicts(df_full, label_col=label_col)
        if conflict_count > 0:
            logging.warning(
                f"⚠️ Found {conflict_count} label conflicts in full dataset before preprocessing.\n{conflicts.head()}"
            )
            logging.info("Resolving conflicts by removing all conflicting rows...")
            df_full = self.resolve_conflicts(df_full, conflicts, label_col=label_col)
        else:
            logging.info("✅ No label conflicts found in full dataset before preprocessing.")

        # Preprocess both
        df_full = self.preprocess_network_data(
            df_full, label_col=label_col, scale_numeric=scale_numeric, use_ordinal_encoding=True
        )
        df_test = self.preprocess_network_data(
            df_test, label_col=label_col, scale_numeric=scale_numeric, use_ordinal_encoding=True
        )

        # Unpack outputs
        X_full, y_full, _, _, _ = df_full
        X_test, y_test, _, _, _ = df_test

        # Join full for label conflict check
        df_full_combined = X_full.copy()
        df_full_combined[label_col] = y_full

        if check_duplicates:
            logging.info("Duplicate Check:")
            logging.info(f"  Full dataset: {X_full.duplicated().sum()} duplicate rows")
            logging.info(f"  Test dataset: {X_test.duplicated().sum()} duplicate rows")
            # drop duplicates in full dataset
            df_full_combined = df_full_combined.drop_duplicates(subset=X_full.columns, keep="first")
            logging.info(f"  Full dataset after dropping duplicates: {df_full_combined.shape[0]} rows")
            # reassign X_full and y_full
            X_full = df_full_combined.drop(columns=[label_col])
            y_full = df_full_combined[label_col]

        # Create training set by subtracting test from full
        df_train = pd.concat([X_full.assign(_label=y_full), X_test.assign(_label=y_test), X_test.assign(_label=y_test)])
        df_train = df_train.drop_duplicates(keep=False)

        y_train = df_train["_label"]
        X_train = df_train.drop(columns=["_label"])

        X_train = self.convert_to_32bit(X_train)
        X_test = self.convert_to_32bit(X_test)

        y_train = y_train.astype(np.int32)
        y_test = y_test.astype(np.int32)

        logging.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        logging.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

        if self.save_processed:
            X_train.to_csv(Path(self.save_path).joinpath("X_train.csv"), index=False)
            X_test.to_csv(Path(self.save_path).joinpath("X_test.csv"), index=False)
            y_train.to_csv(Path(self.save_path).joinpath("y_train.csv"), index=False)
            y_test.to_csv(Path(self.save_path).joinpath("y_test.csv"), index=False)

            # write metadata to json
            self.metadata = {
                **self.metadata,
                **{
                    "num_features": X_train.shape[1],
                    "num_train_samples": X_train.shape[0],
                    "num_test_samples": X_test.shape[0],
                    "feature_names": X_train.columns.tolist(),
                    "label_col": label_col,
                    "cat_cols": self.cat_cols or [],
                    "num_cols": self.num_cols or [],
                },
            }

            with open(Path(self.save_path).joinpath("metadata.json"), "w") as f:
                json.dump(self.metadata, f, indent=2)

            logging.info(f"Processed data saved to {self.save_path}")

        return X_train, X_test, y_train, y_test, X_train.columns.tolist(), self.cat_cols or [], self.num_cols or []

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

        # load metadata
        metadata_path = Path(data_path).joinpath("metadata.json")
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            logging.info(f"Loaded metadata: {metadata}")
        else:
            logging.warning(f"No metadata found at {metadata_path}, using default feature names.")
            metadata = {
                "feature_names": X_train.columns.tolist(),
                "cat_cols": [],
                "num_cols": [],
            }

        self.cat_cols = metadata.get("cat_cols", [])
        self.num_cols = metadata.get("num_cols", [])
        self.metadata = metadata

        return X_train, X_test, y_train, y_test, X_train.columns.tolist(), self.cat_cols, self.num_cols

    @staticmethod
    def check_label_conflicts(df, label_col="type"):
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

    @staticmethod
    def resolve_conflicts(df, conflict_rows, label_col="label"):
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
    X_train, X_test, y_train, y_test, feature_names, cat_cols, num_cols = preprocessor.get_ton_iot_network_data(
        label_col="type",
        scale_numeric=True,
        check_duplicates=False,
        try_load_preprocessed=False,
        drop_labels=["mitm", "ransomware"],
    )

    print(f"Features: {feature_names}")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
