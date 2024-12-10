import logging
from typing import Tuple, List, Optional

import numpy as np
from tensorflow.keras.datasets import imdb
from sklearn.model_selection import train_test_split


word_to_idx = imdb.get_word_index()


def load_imdb(
    num_words: int,
    max_len: Optional[int] = None,
    seed: int = 113,
    val_split: bool = False,
    val_size: float = 0.2,
    verbose: bool = True
) -> Tuple[Tuple[np.ndarray, np.ndarray]]:
    (X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = imdb.load_data(num_words=num_words, maxlen=max_len, seed=seed)

    if val_split:
        np.random.seed(seed)
        X_train_raw, X_valid_raw, y_train_raw, y_valid_raw = train_test_split(
            X_train_raw,
            y_train_raw,
            test_size=val_size,
            random_state=seed
        )
    else:
        X_valid_raw = None
        y_valid_raw = None

    if verbose:
        logging.info(f"num train: {len(X_train_raw)}")
        logging.info(f"num valid: {len(X_valid_raw)}")
        logging.info(f"num test: {len(X_test_raw)}")

    return (X_train_raw, y_train_raw), (X_valid_raw, y_valid_raw), (X_test_raw, y_test_raw)
