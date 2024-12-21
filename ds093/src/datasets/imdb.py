from typing import Tuple, Optional

import numpy as np
from tensorflow.keras.datasets import imdb

from src.utils.data import get_train_valid_test


word_to_idx = imdb.get_word_index()


def load_imdb(
    num_words: int,
    max_len: Optional[int] = None,
    seed: int = 123,
    val_split: bool = False,
    val_size: float = 0.2,
    verbose: bool = True
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    (X_train, y_train), (X_test, y_test) = imdb.load_data(
        num_words=num_words,
        maxlen=max_len, seed=seed
    )

    return get_train_valid_test(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        seed=seed,
        val_split=val_split,
        val_size=val_size,
        verbose=verbose
    )
