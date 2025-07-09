import logging
import json
import pandas as pd

from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report
from src.data_balancing import balance_class_data
from src.model_factory import MODEL_FACTORY, set_seed
from src.pre_processing import GenericPreProcessor, ToNIoTPreProcessor
from src.util import load_yaml, dump_yaml


class ModelTrainer:
    def __init__(self, model_type: str, model_name: str, config: dict, model=None):
        """
        Initialize the ModelTrainer with a specific model.

        Parameters
        ----------
        model_type : str
            Type of the model from MODEL_FACTORY (e.g., 'rf', 'xgb', 'tf_mlp', etc.).
        model_name : str
            Name of the model.
        config : dict, optional
            Configuration parameters for the model, if any.
        model : object, optional
            The model instance to be trained.
        """
        self.model_type = model_type
        self.model_name = model_name
        self.config = config if config is not None else {}

        if model is not None:
            self.model = model
        else:
            if model_type not in MODEL_FACTORY:
                raise ValueError(
                    f"Model type '{model_type}' is not supported. Choose from {list(MODEL_FACTORY.keys())}."
                )

            init_params = self.config.get("init_params", {})
            if model_type in ["mlp", "tf_mlp"]:
                self.model = MODEL_FACTORY[model_type](
                    init_params["num_cont_features"],
                    init_params["cat_dims"],
                    init_params["embed_dims"],
                    init_params["num_classes"],
                    init_params["feature_names"],
                    init_params["cat_cols"],
                )
            elif model_type == "tabnet":
                input_dim = init_params["num_cont_features"] + len(init_params["cat_cols"])
                self.model = MODEL_FACTORY[model_type](input_dim, init_params["num_classes"])
            else:
                self.model = MODEL_FACTORY[model_type]

    def prepare_inputs(self, X: pd.DataFrame) -> tuple:
        """
        Prepare model-specific inputs based on config.

        Returns
        -------
        tuple : Can be (X_cont, X_cat) or (X,) depending on model_type.
        """
        if self.model_type in ["mlp", "tf_mlp"]:
            X_cont = X[self.config["init_params"]["num_cols"]]
            X_cat = X[self.config["init_params"]["cat_cols"]].astype("int32")
            return X_cont, X_cat
        elif self.model_type == "tabnet":
            return X  # tabnet takes full input
        else:
            return X  # sklearn-style

    def train(self, *args, **kwargs):
        """
        Train the model on the provided training data.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features.
        y_train : pd.Series
            Training labels.
        """
        self.model.fit(*args, **kwargs)

        # add in fit parameters to config
        self.config["fit_params"] = {
            "epochs": kwargs.get("epochs"),
            "batch_size": kwargs.get("batch_size"),
            "lr": kwargs.get("lr"),
        }

    def predict(self, *args, **kwargs):
        """
        Make predictions using the trained model.

        Parameters
        ----------
        X_test : pd.DataFrame
            Test features.

        Returns
        -------
        np.ndarray
            Predicted labels.
        """
        return self.model.predict(*args, **kwargs)

    def save(self, model_dir: str):
        """
        Save the trained model to the specified directory.

        Parameters
        ----------
        model_dir : str
            Directory path to save the model.
        """
        # if the model is a TensorFlow model, save it using the Keras save method
        if self.model_type == "tf_mlp":
            self.model.save(f"{model_dir}/{self.model_name}.keras")
            logging.info(f"Model '{self.model_name}' saved to {model_dir}/{self.model_name}.keras")
        # if the model is a PyTorch model, save it using torch.save
        elif self.model_type == "tabnet":
            import torch

            model_path = Path(model_dir) / f"{self.model_name}.pt"
            torch.save(self.model, model_path)
            logging.info(f"Model '{self.model_name}' saved to {model_path}")
        # for other models, use joblib to save the model
        else:
            from joblib import dump

            Path(model_dir).mkdir(parents=True, exist_ok=True)
            dump(self.model, f"{model_dir}/{self.model_name}.joblib")
            logging.info(f"Model '{self.model_name}' saved to {model_dir}/{self.model_name}.joblib")

        self.config["model_type"] = self.model_type
        self.config["model_name"] = self.model_name

        # Save the model configuration to a YAML file
        config_path = Path(model_dir) / f"{self.model_name}_config.yml"
        dump_yaml(self.config, config_path)

    @classmethod
    def load(cls, model_path: str):
        """
        Load a trained model from the specified path.

        Parameters
        ----------
        model_path : str
            Path to the saved model file.
        """
        # if the model is a TensorFlow model, load it using the Keras load method
        if model_path.endswith(".keras"):
            from tensorflow import keras

            model = keras.models.load_model(model_path)
            logging.info(f"Model loaded from {model_path}")
        # if the model is a PyTorch model, load it using torch.load
        elif model_path.endswith(".pt"):
            import torch

            model = torch.load(model_path)
            logging.info(f"Model loaded from {model_path}")
        # for other models, use joblib to load the model
        else:
            from joblib import load

            model = load(model_path)
            logging.info(f"Model loaded from {model_path}")

        # Load the model configuration from the YAML file
        model_file_path = Path(model_path)
        config_path = model_file_path.with_name(f"{model_file_path.stem}_config.yml")
        if config_path.exists():
            config = load_yaml(config_path)
            model_type = config.get("model_type", "unknown")
            model_name = config.get("model_name", "unknown")

        return cls(model_type=model_type, model_name=model_name, model=model, config=config)


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    poisoned_test_data: pd.DataFrame = None,
    poisoned_test_labels: pd.Series = None,
    config_dict: dict = None,
):
    """
    Train and evaluate multiple classifiers on the given tabular dataset.

    Parameters
    ----------
    X_train, X_test : pandas.DataFrame
        Feature matrices.
    y_train, y_test : pandas.Series
        Labels.
    poisoned_test_data, poisoned_test_labels : pandas.DataFrame, pandas.Series, optional
        Poisoned test data and labels for evaluation, if available.
    config_dict : dict, optional
        Configuration dictionary containing model parameters and training settings.

    Returns
    -------
    results : dict
        Dictionary of model names to their accuracy scores.
    """
    cat_cols = config_dict.get("cat_cols", [])
    num_cols = config_dict.get("num_cols", [])
    save_dir = config_dict.get("save_dir", "models")

    if poisoned_test_data is not None and poisoned_test_labels is not None:
        X_poisoned_test_cont = poisoned_test_data[num_cols]
        X_poisoned_test_cat = poisoned_test_data[cat_cols].astype("int32")
        y_poisoned_test = poisoned_test_labels

    cat_dims = [int(X_train[col].max() + 1) for col in cat_cols]
    embed_dims = [min(50, int(round(dim**0.25))) for dim in cat_dims]
    num_cont_features = len(num_cols)
    num_classes = y_train.max() - y_train.min() + 1
    feature_names = X_train.columns.tolist()

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    results = {}
    for model_type, config in config_dict["models_to_train_config"].items():
        if model_type not in MODEL_FACTORY:
            raise ValueError(f"Model '{model_type}' is not supported. Choose from {list(MODEL_FACTORY.keys())}.")

        init_params = {
            "num_cont_features": int(num_cont_features),
            "cat_dims": list(map(int, cat_dims)),
            "embed_dims": list(map(int, embed_dims)),
            "num_classes": int(num_classes),
            "feature_names": list(feature_names),
            "cat_cols": list(cat_cols),
            "num_cols": list(num_cols),
        }
        training_config = {}
        training_config["init_params"] = init_params

        print(f"Training: {model_type.upper()}")
        if model_type == "tf_mlp":
            training_config["fit_params"] = config.get("fit_params", {})
            training_config["fit_params"]["X_val_cont"] = X_test[num_cols]
            training_config["fit_params"]["X_val_cat"] = X_test[cat_cols].astype("int32")
            training_config["fit_params"]["y_val"] = y_test

            trainer = ModelTrainer(
                model_type="tf_mlp",
                model_name=f"{model_type}_model",
                config=training_config,
            )

            X_train_cont, X_train_cat = trainer.prepare_inputs(X_train)
            X_test_cont, X_test_cat = trainer.prepare_inputs(X_test)
            trainer.train(X_train_cont, X_train_cat, y_train, **training_config["fit_params"])

            preds = trainer.predict(X_test_cont, X_test_cat)
            if poisoned_test_data is not None and poisoned_test_labels is not None:
                X_poisoned_test_cont, X_poisoned_test_cat = trainer.prepare_inputs(poisoned_test_data)
                poisoned_preds = trainer.predict(X_poisoned_test_cont, X_poisoned_test_cat)
                poisoned_acc = accuracy_score(y_poisoned_test, poisoned_preds)
                print(f"Poisoned Test Accuracy: {poisoned_acc:.4f}")
        else:
            model_name = f"{model_type}_model"
            trainer = ModelTrainer(
                model_type=model_type,
                model_name=model_name,
                config=training_config,
            )

            trainer.train(X_train, y_train)
            preds = trainer.predict(X_test)
            if poisoned_test_data is not None and poisoned_test_labels is not None:
                poisoned_preds = trainer.predict(poisoned_test_data)
                poisoned_acc = accuracy_score(poisoned_test_labels, poisoned_preds)
                print(f"Poisoned Test Accuracy: {poisoned_acc:.4f}")
        acc = accuracy_score(y_test, preds)
        results[model_type] = acc

        # dump accuracy and classification report to a file
        with open(f"{save_dir}/{model_type}_results.txt", "w") as f:
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(classification_report(y_test, preds, digits=3))
            if poisoned_test_data is not None and poisoned_test_labels is not None:
                f.write(f"Poisoned Test Accuracy: {poisoned_acc:.4f}\n")
                f.write(classification_report(y_poisoned_test, poisoned_preds, digits=3))

        if model_type == "xgb":
            # plot feature importances for XGBoost
            import matplotlib.pyplot as plt
            import xgboost as xgb

            xgb.plot_importance(trainer.model, importance_type="gain")
            plt.title("Feature Importances")
            plt.tight_layout()
            plt.savefig(f"{save_dir}/{model_type}_feature_importances.png")

            # print feature importances
            feature_importances = pd.Series(trainer.model.feature_importances_, index=X_train.columns)
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
        disp.ax_.set_title(f"Confusion Matrix for {model_type.upper()}")
        plt.savefig(f"{save_dir}/{model_type}_confusion_matrix.png")

        if poisoned_test_data is not None and poisoned_test_labels is not None:
            disp_poisoned = ConfusionMatrixDisplay.from_predictions(
                y_poisoned_test,
                poisoned_preds,
                normalize="true",
            )
            disp_poisoned.ax_.set_title(f"Poisoned Test Confusion Matrix for {model_type.upper()}")
            plt.savefig(f"{save_dir}/{model_type}_poisoned_confusion_matrix.png")

        if config.get("save_model", True):
            trainer.save(save_dir)

    return results


def load_data(smote_data_dir: Path):
    smote_train_data = smote_data_dir / "smote_train_data.csv"
    smote_train_labels = smote_data_dir / "smote_train_labels.csv"
    test_data = smote_data_dir / "test_data.csv"
    test_labels = smote_data_dir / "test_labels.csv"

    if not smote_train_data.exists() or not smote_train_labels.exists():
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
        X_train, y_train = balance_class_data(
            X_train, y_train, method="SMOTENC", reduce_majority=True, cat_cols=cat_cols
        )
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

    # Round data to avoid floating point issues with categorical data
    X_train = X_train.round(7).astype("float32")
    X_test = X_test.round(7).astype("float32")

    # load metadata from json file
    with open(smote_data_dir.joinpath("metadata.json"), "r") as f:
        metadata = json.load(f)

    return X_train, y_train, X_test, y_test, metadata


def load_models(models_to_load: list, model_dir: str = "models"):
    models = {}
    for model_type in models_to_load:
        if model_type in ["rf", "xgb"]:
            model_path = model_dir / f"{model_type}_model.joblib"
        elif model_type == "tf_mlp":
            model_path = model_dir / f"{model_type}_model.keras"
        elif model_type == "tabnet":
            model_path = model_dir / f"{model_type}_model.pt"
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        models[model_type] = ModelTrainer.load(model_path)

    return models


def prepare_data_for_training(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, config_dict: dict):
    """Prepares the training and testing data for model training."""
    norm_cols = config_dict.get("num_cols", [])
    X_train, y_train = GenericPreProcessor.check_and_resolve_label_conflicts(X_train, y_train)
    X_train_norm, scaler = GenericPreProcessor.normalize_data(X_train, scale_numeric=True, columns=norm_cols)
    X_test_norm, _ = GenericPreProcessor.normalize_data(
        X_test, scale_numeric=True, existing_scaler=scaler, columns=norm_cols
    )
    return X_train_norm, y_train, X_test_norm, scaler


if __name__ == "__main__":
    set_seed(42)
    DEBUG = True
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

    config = load_yaml("configs/model_training_config.yml")

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

    if train_clean:
        X_train_norm, y_train, X_test_norm, _ = prepare_data_for_training(X_train, y_train, X_test, config_dict=config)

        logging.info("Training and testing data loaded successfully.")
        # Train models and evaluate
        results = train_models(
            X_train_norm,
            y_train,
            X_test_norm,
            y_test,
            config_dict=config,
        )
        print("Training results:", results)

    # Poison data and retrain
    if poison_fraction is None:
        logging.info("No poison fraction specified. Skipping backdoor poisoning.")
        exit(0)

    from src.backdoor_attack import BackdoorPoisoner

    poisoner = BackdoorPoisoner(
        trigger_fn=BackdoorPoisoner.all_trigger,
        target_label=metadata["label_mapping"]["normal"],  # Target label for poisoned samples
    )

    logging.info(f"Poisoning {poison_fraction * 100:.2f}% of the training data.")

    X_poisoned, y_poisoned = poisoner.poison(X_train, y_train, poison_fraction=poison_fraction, random_state=42)
    X_poisoned_norm, y_posoned, X_test_norm, scaler = prepare_data_for_training(
        X_poisoned, y_poisoned, X_test, config_dict=config
    )

    # poison test data
    X_poisoned_test, y_poisoned_test = poisoner.poison(X_test, y_test, poison_fraction=1.0, random_state=42)
    X_poisoned_test, _ = GenericPreProcessor.normalize_data(
        X_poisoned_test, scale_numeric=True, existing_scaler=scaler, columns=norm_cols
    )

    results = train_models(
        X_poisoned_norm,
        y_poisoned,
        X_test_norm,
        y_test,
        poisoned_test_data=X_poisoned_test,
        poisoned_test_labels=y_poisoned_test,
        config_dict=config,
    )
    print("Training results (after poisoning):", results)
