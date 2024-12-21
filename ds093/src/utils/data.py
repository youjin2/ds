import logging
from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split


def get_train_valid_test(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int = 123,
    val_split: bool = False,
    val_size: float = 0.2,
    verbose: bool = True,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:

    # train/valid split
    if val_split:
        np.random.seed(seed)
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train,
            y_train,
            test_size=val_size,
            random_state=seed
        )
    else:
        X_valid = []
        y_valid = []

    if verbose:
        logging.info(f"num train: {len(X_train)}")
        logging.info(f"num valid: {len(X_valid)}")
        logging.info(f"num test: {len(X_test)}")

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)
