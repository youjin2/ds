from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.data import get_train_valid_test


def load_dataframe() -> pd.DataFrame:
    df = pd.read_csv(
        "https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/train.tsv",
        delimiter="\t",
        header=None,
        names=["review", "sentiment"],
    )
    return df


def load_sst2(
    num_sample: int = None,
    seed: int = 123,
    val_split: bool = False,
    val_size: float = 0.2,
    test_size: float = 0.2,
    verbose: bool = True,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    df = load_dataframe()

    # use num_sample data
    if num_sample:
        df = df.sample(min(num_sample, len(df)), random_state=seed)

    # train/test split
    review = df.review.values
    sentiment = df.sentiment.values

    X_train, X_test, y_train, y_test = train_test_split(
        review,
        sentiment,
        test_size=test_size,
        random_state=seed
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
