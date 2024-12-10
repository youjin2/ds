from typing import Tuple

import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import History


def plot_history(
    history: History,
    line_type: str = "loss",
    ylabel: str = "Loss",
) -> Tuple[plt.Figure, plt.Axes]:
    train_loss = history.history[f"{line_type}"]
    valid_loss = history.history[f"val_{line_type}"]

    fig, ax = plt.subplots(1, 1)
    ax.plot(train_loss, label="Train")
    ax.plot(valid_loss, label="Valid")
    ax.grid(linestyle=":")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend()

    return (fig, ax)
