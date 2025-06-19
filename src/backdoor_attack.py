import pandas as pd
import numpy as np
from typing import Tuple, Callable, Union


class BackdoorPoisoner:
    def __init__(
        self,
        trigger_fn: Callable[[pd.DataFrame], pd.DataFrame],
        target_label: int,
    ):
        """
        Custom backdoor poisoner without using ART.

        Parameters
        ----------
        trigger_fn : Callable
            Function that applies the backdoor trigger to selected samples.
        target_label : int
            The label assigned to poisoned samples.
        """
        self.trigger_fn = trigger_fn
        self.target_label = target_label
        self.poison_indices_ = None  # For auditing

    @staticmethod
    def dns_trigger(X: np.ndarray, columns: list[str]) -> np.ndarray:
        # find the index of the columns
        col_indices = {col: i for i, col in enumerate(columns)}

        X[:, col_indices["dns_RD"]] = 1
        X[:, col_indices["dns_RA"]] = 1
        X[:, col_indices["dns_rejected"]] = 1
        return X
    
    @staticmethod
    def bytes_trigger(X: np.ndarray, columns: list[str]) -> np.ndarray:
        # find the index of the columns
        col_indices = {col: i for i, col in enumerate(columns)}

        X[:, col_indices["src_bytes"]] = 1000
        X[:, col_indices["src_ip_bytes"]] = 50
        X[:, col_indices["dst_bytes"]] = 1045
        X[:, col_indices["dst_ip_bytes"]] = 94
        return X
    
    @staticmethod
    def all_trigger(X: np.ndarray, columns: list[str]) -> np.ndarray:
        X = BackdoorPoisoner.dns_trigger(X, columns)
        X = BackdoorPoisoner.bytes_trigger(X, columns)
        return X

    def poison(
        self,
        X_train: Union[pd.DataFrame],
        y_train: Union[pd.Series],
        poison_fraction: float = 0.05,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Poison a fraction of training data.

        Returns
        -------
        X_poisoned, y_poisoned : np.ndarray, np.ndarray
            Features and labels with poisoned samples included.
        """
        if isinstance(X_train, pd.DataFrame):
            feature_names = X_train.columns
            X_train = X_train.values
        else:
            feature_names = None

        if isinstance(y_train, pd.Series):
            y_train = y_train.values

        np.random.seed(random_state)

        # Select only indices where the label is NOT the target label
        non_target_indices = np.where(y_train != self.target_label)[0]
        n_poison = min(int(poison_fraction * len(y_train)), len(non_target_indices))

        # Randomly choose from non-target indices
        poison_indices = np.random.choice(non_target_indices, size=n_poison, replace=False)
        self.poison_indices_ = poison_indices

        X_poisoned = X_train.copy()
        y_poisoned = y_train.copy()

        # Apply trigger only to selected samples
        X_poisoned[poison_indices] = self.trigger_fn(X_poisoned[poison_indices], feature_names)
        y_poisoned[poison_indices] = self.target_label

        # Optionally restore DataFrame
        if feature_names is not None:
            X_poisoned = pd.DataFrame(X_poisoned, columns=feature_names)

        return X_poisoned, y_poisoned
