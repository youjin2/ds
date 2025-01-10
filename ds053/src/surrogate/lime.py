from typing import Callable, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from skimage.color import label2rgb
from lime.lime_text import LimeTextExplainer
from lime.lime_image import LimeImageExplainer
from lime.explanation import Explanation
from lime.wrappers.scikit_image import SegmentationAlgorithm


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


class LimeImageSummarizer:
    def __init__(
        self,
        explainer: LimeImageExplainer,
        segmentation_fn: SegmentationAlgorithm,
        classifier_fn: Callable,
    ) -> None:
        self.explainer = explainer
        self.segmentation_fn = segmentation_fn
        self.classifier_fn = classifier_fn

    def set_data(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> None:
        assert len(X) == len(y), "length of X and y must be equivalent"
        assert len(X.shape) == 4, "dim(X) must be 4-dimensional (batch_size, height, width, channel)"

        self._X = X
        self._y = y
        self._n = len(X)

    def __call__(
        self,
        idx: int,
        top_labels: int = 6,
        num_samples: int = 1000,
        num_features: int = 8,
        **kwargs,
    ) -> Tuple[Explanation, plt.Figure]:
        if idx < 0 or idx >= self._n:
            raise IndexError(f"idx must be between [0, {self._n-1}")

        explanation = self.explainer.explain_instance(
            self._X[idx],
            classifier_fn=self.classifier_fn,
            top_labels=top_labels,
            num_samples=num_samples,
            segmentation_fn=self.segmentation_fn
        )

        tmp_pos, mask_pos = explanation.get_image_and_mask(
            self._y[idx],
            positive_only=True,
            num_features=num_features,
            hide_rest=kwargs.get("hide_rest", False)
        )

        tmp_all, mask_all = explanation.get_image_and_mask(
            self._y[idx],
            positive_only=False,
            num_features=num_features,
            hide_rest=kwargs.get("hide_rest", False)
        )

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        ax1.imshow(label2rgb(mask_pos, tmp_pos, bg_label=0))
        ax1.set_title(f"Positive Regions for {self._y[idx]}")

        ax2.imshow(label2rgb(4-mask_all, tmp_all, bg_label=0))
        ax2.set_title(f"Positive/Negative Regions for {self._y[idx]}")

        ax3.imshow(tmp_all)
        ax3.set_title("Output Image Only")

        ax4.imshow(mask_all)
        ax4.set_title("Mask Only")

        for ax in [ax1, ax2, ax3, ax4]:
            ax.axis("off")

        return (explanation, fig)
