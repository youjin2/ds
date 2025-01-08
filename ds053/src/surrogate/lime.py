from typing import Callable, List

from lime.lime_text import LimeTextExplainer
from lime.explanation import Explanation


class LimeTextSummarizer:
    def __init__(
        self,
        explainer: LimeTextExplainer,
        classifier_fn: Callable,
    ) -> None:
        self.explainer = explainer
        self.classifier_fn = classifier_fn
        self.class_names = explainer.class_names

    def set_data(
        self,
        X: List[str],
        y: List[int],
    ) -> None:
        assert len(X) == len(y), "length of X and y must be equivalent"

        self._X = X
        self._y = y
        self._n = len(X)

    def __call__(
        self,
        idx: int,
        top_labels: int = 1,
        num_display_labels: int = 1,
        **kwargs,
    ) -> Explanation:
        if idx < 0 or idx >= self._n:
            raise IndexError(f"idx must be between [0, {self._n-1}]")

        input_text = self._X[idx]
        true_label = self._y[idx]

        pred_probs = self.classifier_fn([input_text])[0]
        ranks = sorted(range(len(pred_probs)), key=lambda i: pred_probs[i], reverse=True)
        pred_label = ranks[0]

        print("====="*10)
        print(f"Document ID: {idx}")
        print("====="*10)
        print(f"True Label: {true_label} ({self.class_names[true_label]})")
        print(f"Pred Label: {pred_label} ({self.class_names[pred_label]})")
        print()
        for rank in ranks:
            cate = self.class_names[rank]
            prob = pred_probs[rank]
            print(f"Category: {cate:>15} ({prob*100:.2f}%)")

        print()
        explanation = self.explainer.explain_instance(
            input_text,
            classifier_fn=self.classifier_fn,
            top_labels=top_labels,
        )
        available_labels = explanation.available_labels()
        explanation.show_in_notebook(text=input_text, labels=available_labels[:num_display_labels])

        return explanation
