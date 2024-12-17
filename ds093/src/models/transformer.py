from numpy import arange
import tensorflow as tf


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position: tf.Tensor, d_model: int) -> None:
        super().__init__()
        self.pos_encoding = self.encode()

    def get_angles(self, position, d_model: int) -> tf.Tensor:
        num = position
        denom = tf.pow(
            tf.constant(10000., dtype="float32")
            tf.divide(
                2 * (tf.range(d_model, dtype="float32")[tf.newaxis, ...] // 2),
                tf.constant(d_model, dtype="float32")
            )
        )
        return tf.divide(num, denom)

    def encode(self, position: tf.Tensor, d_model: int) -> tf.Tensor:
        pass

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        pass


class Encoder(tf.keras.layers.Layer):
    def __init__(self) -> None:
        pass

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        pass


class Decoder(tf.keras.layers.Layer):
    def __init__(self) -> None:
        pass

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        pass




    
