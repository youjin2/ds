import gc

import tensorflow as tf


def reset_session() -> None:
    # to prevent memory leak
    tf.keras.backend.clear_session()
    gc.collect()
