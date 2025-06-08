import numpy as np
import pandas as pd
from art.attacks.poisoning import PoisoningAttackBackdoor
from art.estimators.classification import SklearnClassifier, PyTorchClassifier
from typing import Union, Callable, Tuple


class ARTBackdoorPoisoner:
    def __init__(
        self,
        trigger_fn: Callable[[np.ndarray], np.ndarray],
        target_label: int,
        classifier: Union[SklearnClassifier, PyTorchClassifier],
    ):
        """
        Initialize the backdoor poisoner using ART.

        Parameters
        ----------
        trigger_fn : Callable
            A function that modifies a batch of samples by inserting the backdoor trigger.
        target_label : int
            Label to assign to the poisoned samples.
        classifier : SklearnClassifier or PyTorchClassifier
            ART-wrapped classifier.
        """
        self.trigger_fn = trigger_fn
        self.target_label = target_label
        self.classifier = classifier
        self.attack = PoisoningAttackBackdoor(
            backdoor=self.trigger_fn,
            target=self.target_label,
            estimator=self.classifier,
        )

    def poison(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        poison_fraction: float = 0.05,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Poison a fraction of training data using the backdoor attack.

        Parameters
        ----------
        X_train : np.ndarray or pd.DataFrame
            Clean training features.
        y_train : np.ndarray or pd.Series
            Clean training labels.
        poison_fraction : float
            Fraction of training samples to poison.
        random_state : int
            Seed for reproducibility.

        Returns
        -------
        X_poisoned : np.ndarray
            Training features with poisoned samples.
        y_poisoned : np.ndarray
            Corresponding labels.
        """
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values

        np.random.seed(random_state)
        return self.attack.poison(X_train, y_train, poison_percent=poison_fraction)
