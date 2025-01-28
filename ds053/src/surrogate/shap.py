from copy import deepcopy
from math import factorial
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
            # replace feature indices in train_data whose x value is not nan with the value of x
            data[:, ~nan_idxs] = x[~nan_idxs]
            pred = np.mean(self.model.predict(data))

        return pred


class ExactShap:
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

    def subsets(self, idxs: Set[int]) -> List[List[int]]:
        idxs = list(idxs)
        n = len(idxs)
        result = []
        # 2^n possible subsets
        for i in range(1 << n):
            cur = [idxs[j] for j in range(n) if (i & (1 << j))]
            result.append(cur)
        return result

    def eval_f_s(
        self,
        x: np.ndarray,
        subset_idxs: List[int],
    ) -> float:
        data = deepcopy(self.data)
        model = self.model

        # if none of features are choosen, return the average value of train data
        if len(subset_idxs) == 0:
            return np.mean(model.predict(data))
        # if all features are choosen, return f(x)
        elif len(subset_idxs) == data.shape[1]:
            x = np.expand_dims(x, axis=0)
            return np.mean(model.predict(x))
        # if some feature are choosen, replace feature values whose indices are not choosen with the train data
        else:
            data[:, subset_idxs] = x[subset_idxs]
            return np.mean(model.predict(data))

    def shap_values(self, x: np.ndarray) -> np.ndarray:
        if len(x.shape) != 1:
            raise ValueError(
                "x must be 1-dimensional, x=(x_1, ..., x_p)" +
                " where p is the number of input features."
            )

        nan_idxs_set = set(np.where(np.isnan(x))[0])
        phi = []
        for i, val in enumerate(x):
            if np.isnan(val):
                phi.append(0.)
            else:
                # the number of features
                p = self.data.shape[1]
                target_set = set(range(p)) - {i}

                # exception for nan features
                if nan_idxs_set:
                    target_set = target_set - nan_idxs_set
                    p -= len(nan_idxs_set)

                # calculates shapley value for feature i
                all_subset_idxs = self.subsets(target_set)
                cur_phi = 0
                for subset_idxs in all_subset_idxs:
                    s = len(subset_idxs)
                    f_si = self.eval_f_s(x, subset_idxs + [i])
                    f_s = self.eval_f_s(x, subset_idxs)
                    cur_phi += (factorial(s) * factorial(p - s - 1) * (f_si - f_s))

                # normalize with the number of all possible cases
                cur_phi /= factorial(p)

                phi.append(cur_phi)

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
            # replace feature indices in train_data whose x value is not nan with the value of x
            data[:, ~nan_idxs] = x[~nan_idxs]
            pred = np.mean(self.model.predict(data))

        return pred
