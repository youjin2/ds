import logging
from typing import Tuple

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class FilterVisualizer:
    def __init__(
        self,
        model: tf.keras.models.Model
    ) -> None:
        self.model = model

    def _set_submodel(self, layer_name: str) -> None:
        """
        set submodel returning given layer_name as output
        """
        inputs = self.model.input
        try:
            outputs = self.model.get_layer(layer_name).output
            self.submodel = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        except ValueError as e:
            # names = [l.name for l in self.model.layers]
            logging.error(e)

    def _get_submodel_output(self, image: np.ndarray) -> np.ndarray:
        """
        return submodel's output
        """
        assert len(image.shape) == 3, "Input image must be 3-dimensional (height, width, channel)"
        image = np.expand_dims(image, axis=0)
        output = self.submodel(image)[0].numpy()
        vmin = np.min(output, axis=(0, 1))
        vmax = np.max(output, axis=(0, 1))
        output = ((output - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        return output

    def __call__(
        self,
        image: np.ndarray,
        num_cols: int = 10,
        top_k: int = 5,
        verbose: bool = True,
        **kwargs,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        visualize filters of submodel's output
        """
        assert len(image.shape) == 3, "Input image must be 3-dimensional (height, width, channel)"

        output = self._get_submodel_output(image)
        num_channels = output.shape[-1]
        q, r = divmod(num_channels, num_cols)
        num_rows = q + 1 if r > 0 else q

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5))
        for i in range(num_channels):
            rn, cn = divmod(i, num_cols)
            if len(axes.shape) == 2:
                ax = axes[rn, cn]
            else:
                ax = axes[cn]
            cur = output[..., i]
            ax.imshow(cur, cmap="gray")
            ax.set_title(f"Filterta {i}")
            ax.axis("off")

        # turn off remaining axis
        for ax in axes.flatten():
            if not ax.title.get_text():
                ax.axis("off")

        # print top-k labels with the highest probability
        if verbose:
            pred_probs = self.model.predict(np.expand_dims(image, axis=0), verbose=0)[0]
            labels = sorted(range(len(pred_probs)), key=lambda i: pred_probs[i], reverse=True)
            for i in range(top_k):
                print(f"Label: {labels[i]} (Prob: {pred_probs[labels[i]]*100:.4f}%)")

        return (fig, axes)
