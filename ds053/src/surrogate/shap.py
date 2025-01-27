from copy import deepcopy
from typing import List, Set

import numpy as np
from sklearn.linear_model import LinearRegression


class LinearShap:
    def __init__(
        self,
        model: LinearRegression,
        data: np.ndarray,
    ) -> None:
        self.model = model
        self.data = data
        # E[f(X)], X: train data (or subset)
        self.__phi0 = np.mean(self.model.predict(data))

    @property
    def expected_value(self) -> float:
        return self.__phi0

    def shap_values(self, x: np.ndarray) -> np.ndarray:
        if len(x.shape) != 1:
            raise ValueError(
                "x must be 1-dimensional, x=(x_1, ..., x_p)" +
                " where p is the number of input features."
            )

        phi = []
        for i, val in enumerate(x):
            if np.isnan(val):
                phi.append(0.)
            else:
                # phi_i = b_i*(x_i - E[X_i]), X_i: i-th feature values in train data
                phi.append(self.model.coef_[i]*(val - np.mean(self.data[:, i])))

        return np.array(phi)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if len(x.shape) != 1:
            raise ValueError(
                "x must be 1-dimensional, x=(x_1, ..., x_p)" +
                " where p is the number of input features."
            )

        if not np.any(np.isnan(x)):
            pred = self.model.predict(np.expand_dims(x, axis=0))[0]
        else:
            data = deepcopy(self.data)
            nan_idxs = np.isnan(x)
            # replace feature values in train_data with non-nan feature values in given data point x
            data[:, ~nan_idxs] = x[~nan_idxs]
            pred = np.mean(self.model.predict(data))

        return pred
