import pandas as pd
import numpy as np
import logging

from typing import Optional
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer


class ToNIoTPreProcessor:
    def __init__(self, processed_data_path: str, test_data_path: str):
        self.processed_data_path = Path(processed_data_path).as_posix()
        self.test_data_path = Path(test_data_path).as_posix()
        self.column_transformer = None

    def _load_data(self, csv_path: str) -> pd.DataFrame:
        """
        Load the ToN-IoT data from the specified csv path.
        """
        try:
            df = pd.read_csv(csv_path)
            logging.info(f"Loaded processed data from {csv_path}")
            return df
        except FileNotFoundError as e:
            logging.error(f"Processed data file not found at {csv_path}")
            raise e

    def pre_process_iot(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess IoT telemetry dataframe (e.g., fridge dataset) by converting date/time fields
        into useful features and standardizing the format.

        Parameters
        ----------
        df : pandas.DataFrame
            Raw dataframe containing at least 'date' and 'time' columns.

        Returns
        -------
        df : pandas.DataFrame
            Processed dataframe with additional time-based features and a unified datetime field.
        """
        df["date"] = df["date"].astype(str).str.strip()
        df["time"] = df["time"].astype(str).str.strip()

        df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"], format="%d-%b-%y %H:%M:%S", errors="coerce")
        df = df.dropna(subset=["datetime"])

        # Extract time-based features
        df["hour"] = df["datetime"].dt.hour
        df["dayofweek"] = df["datetime"].dt.dayofweek
        df["month"] = df["datetime"].dt.month

        df = df.drop(columns=["date", "time", "datetime"])

        return df

    def preprocess_network_data(
        self,
        df: pd.DataFrame,
        label_col: str = "label",
        drop_cols: Optional[list[str]] = None,
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
        drop_cols : list of str or None
            Columns to drop (e.g., IPs, timestamps, URI fields).
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
        if drop_cols is None or len(drop_cols) == 0:
            drop_cols = [
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

        df = df.copy()
        df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")
        df = df.dropna()

        df["src_bytes"] = pd.to_numeric(df["src_bytes"], errors="coerce")
        df = df.dropna(subset=["src_bytes"])

        y = df[label_col].astype(int)
        X = df.drop(columns=[label_col])

        # Identify column types
        cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
        num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

        for col in cat_cols:
            types = X[col].map(type).unique()
            if len(types) > 1:
                logging.warning(f"⚠️ Column '{col}' contains mixed types: {types}")

        # Construct transformer
        if self.column_transformer is None:
            if use_ordinal_encoding:
                logging.info("Using Ordinal Encoding for categorical features.")
                cat_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            else:
                logging.info("Using One-Hot Encoding for categorical features.")
                cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)

            self.column_transformer = ColumnTransformer(
                [
                    ("num", StandardScaler() if scale_numeric else "passthrough", num_cols),
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

    def get_ton_iot_IOT_data(
        self,
        label_col: str,
        drop_cols: list[str] = [],
        scale_numeric: bool = True,
        encode_categorical: bool = True,
        check_duplicates: bool = True,
    ):
        df_full = self._load_data(self.processed_data_path)
        df_test = self._load_data(self.test_data_path)

        # Align columns
        df_full = df_full[df_test.columns]
        logging.info(f"Full dataset shape: {df_full.shape}, Test dataset shape: {df_test.shape}")

        df_full = self.pre_process_iot(df_full)
        df_test = self.pre_process_iot(df_test)

        logging.info(f"Full dataset shape: {df_full.shape}, Test dataset shape: {df_test.shape}")

        # Check for label conflicts
        conflicts, conflict_count = self.check_label_conflicts(df_full, label_col)
        if conflict_count > 0:
            logging.warning(f"Found {conflict_count} label conflicts in training data. Conflicting rows:\n{conflicts}")
        else:
            logging.info("No label conflicts found in training data.")

        if check_duplicates:
            logging.info("Duplicate Check:")
            full_dupes = df_full.duplicated().sum()
            test_dupes = df_test.duplicated().sum()
            ts_dupes = df_full.duplicated(subset=["ts"]).sum() if "ts" in df_full.columns else "N/A"

            logging.info(f"  Full dataset: {full_dupes} duplicate rows, shape: {df_full.shape}")
            logging.info(f"  Full dataset: {ts_dupes} rows with duplicate timestamps, shape: {df_full.shape}")
            logging.info(f"  Test dataset: {test_dupes} duplicate rows, shape: {df_test.shape}")

        # Create train set by subtracting test set from full set
        df_train = pd.concat([df_full, df_test]).drop_duplicates(keep=False)

        # Drop unnecessary columns
        df_train = df_train.drop(columns=[col for col in drop_cols if col in df_train.columns])
        df_test = df_test.drop(columns=[col for col in drop_cols if col in df_test.columns])

        # Split labels and features
        y_train = df_train[label_col]
        X_train = df_train.drop(columns=[label_col])
        y_test = df_test[label_col]
        X_test = df_test.drop(columns=[label_col])

        # Process feature types
        cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

        if encode_categorical:
            for col in cat_cols:
                le = LabelEncoder()
                X_train[col] = le.fit_transform(X_train[col].astype(str))
                X_test[col] = le.transform(X_test[col].astype(str))

        if scale_numeric and num_cols:
            scaler = StandardScaler()
            X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
            X_test[num_cols] = scaler.transform(X_test[num_cols])

        float_cols = X_train.select_dtypes(include="float64").columns
        int_cols = X_train.select_dtypes(include="int64").columns

        X_train[float_cols] = X_train[float_cols].astype("float32")
        X_test[float_cols] = X_test[float_cols].astype("float32")
        X_train[int_cols] = X_train[int_cols].astype("int32")
        X_test[int_cols] = X_test[int_cols].astype("int32")

        # Convert to float32 to save memory
        y_train = y_train.astype(np.int32)
        y_test = y_test.astype(np.int32)

        return X_train, X_test, y_train, y_test, X_train.columns.tolist()

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
        df_full = self._load_data(self.processed_data_path)
        df_test = self._load_data(self.test_data_path)

        # Align schema
        df_full = df_full[df_test.columns]
        logging.info(f"Full dataset shape: {df_full.shape}, Test dataset shape: {df_test.shape}")

        # Preprocess both
        df_full = self.preprocess_network_data(
            df_full, label_col=label_col, drop_cols=drop_cols, scale_numeric=scale_numeric
        )
        df_test = self.preprocess_network_data(
            df_test, label_col=label_col, drop_cols=drop_cols, scale_numeric=scale_numeric
        )

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
        "data/ton_iot/Processed_datasets/Processed_Network_dataset/Network_dataset_1.csv",
        "data/ton_iot/Train_Test_datasets/Train_Test_Network_dataset/train_test_network.csv",
        # "data/ton_iot/Processed_datasets/Processed_IoT_dataset/IoT_Fridge.csv",
        # "data/ton_iot/Train_Test_datasets/Train_Test_IoT_dataset/Train_Test_IoT_Fridge.csv",
    )
    X_train, X_test, y_train, y_test, feature_names = preprocessor.get_ton_iot_network_data(
        label_col="label",
        scale_numeric=True,
        check_duplicates=True,
    )

    print(f"Features: {feature_names}")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
